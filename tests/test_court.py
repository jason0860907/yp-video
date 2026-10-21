"""Court calibration: the fit recovers a known camera, and the store and
router round-trip marks without keeping a derived matrix."""
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yp_video.court import annotations as store
from yp_video.court import camera, geometry, positions, trajectory
from yp_video.court.annotations import Calibration
from yp_video.extraction import feet as extraction_feet
from yp_video.web.routers import court as routes

SIZE = (1920, 1080)

# A plausible sideline camera: court metres → normalized image.
CAMERA = np.array([[0.045, 0.012, 0.08], [0.0, -0.032, 0.82], [0.0, 0.022, 1.0]])


def _marks(names):
    return {
        n: tuple(float(v) for v in geometry.project(CAMERA, np.array([geometry.LANDMARKS[n]]))[0])
        for n in names
    }


def test_fit_recovers_the_camera_and_projects_feet():
    fit = geometry.fit(_marks(["left_near", "left_far", "center_near", "center_far", "right_near"]))
    assert fit.rmse_m == pytest.approx(0, abs=1e-6)
    feet = geometry.project(CAMERA, np.array([[4.5, 3.0]]))
    court = geometry.project(np.array(fit.image_to_court), feet)
    assert court[0] == pytest.approx([4.5, 3.0], abs=1e-6)
    back = geometry.project(np.array(fit.court_to_image), court)
    assert back[0] == pytest.approx(feet[0], abs=1e-6)


def test_court_to_image_puts_the_floor_in_front_of_the_camera():
    # A homography is only defined up to scale: the solver may hand back
    # the negated matrix, whose inverse puts the floor behind the horizon.
    solve = geometry.cv2.findHomography
    with patch.object(geometry.cv2, "findHomography", lambda *a: (-solve(*a)[0], None)):
        fit = geometry.fit(_marks(["left_near", "left_far", "center_near", "center_far", "right_near"]))
    corners = np.array([[x, y, 1.0] for x in (0, 18) for y in (0, 9)])
    assert (corners @ np.array(fit.court_to_image)[2] > 0).all()


def test_fit_rejects_too_few_or_collinear_marks():
    with pytest.raises(geometry.FitError, match="at least 4"):
        geometry.fit(_marks(["left_near", "left_far", "center_near"]))
    with pytest.raises(geometry.FitError, match="one line"):
        geometry.fit(_marks(["left_near", "left_attack_near", "center_near", "right_near"]))


def test_net_top_marks_never_enter_the_floor_fit():
    marks = _marks(["left_near", "left_far", "right_near"])
    marks["net_top_near"] = (0.5, 0.3)
    with pytest.raises(geometry.FitError, match="3 marked"):
        geometry.fit(marks)
    marks.update(_marks(["right_far"]))
    assert geometry.fit(marks).rmse_m == pytest.approx(0, abs=1e-6)


def test_fit_reports_disagreement_past_four_points():
    marks = _marks(["left_near", "left_far", "right_near", "right_far", "center_near"])
    marks["center_near"] = (marks["center_near"][0] + 0.03, marks["center_near"][1])
    assert geometry.fit(marks).rmse_m > 0.1


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "COURT_ANNOTATIONS_DIR", tmp_path / "court")
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/court")
    video = tmp_path / "game.mp4"
    with patch.object(routes, "resolve_cut", return_value=video), patch.object(routes, "sync_to_r2"):
        yield TestClient(app)


def test_router_round_trips_marks_and_serves_the_fit(client):
    empty = client.get("/api/court/video/game.mp4").json()
    assert empty["points"] == {} and empty["fit"] is None and "at least 4" in empty["fit_error"]
    assert empty["net_height_m"] == 2.43
    marks = _marks(["left_near", "left_far", "right_near", "right_far"])
    saved = client.put("/api/court/video/game.mp4", json={"points": marks, "frame_size": list(SIZE)}).json()
    assert saved["fit"]["rmse_m"] == pytest.approx(0, abs=1e-6)
    assert set(store.load("game").points) == set(marks)
    assert "image_to_court" not in store.annotation_path("game").read_text()
    assert client.get("/api/court/video/game.mp4").json()["fit"] == saved["fit"]


def test_router_keeps_the_guide_lines(client):
    assert client.get("/api/court/video/game.mp4").json()["guides"] == []
    # A guide may reach past the frame edge, to an off-screen corner.
    guides = [[[0.1, 0.9], [0.6, 0.5]], [[-0.2, 0.7], [0.9, 1.1]]]
    body = {"points": {}, "frame_size": list(SIZE), "guides": guides}
    assert client.put("/api/court/video/game.mp4", json=body).json()["guides"] == guides
    assert client.get("/api/court/video/game.mp4").json()["guides"] == guides
    body["guides"] = [[[0.1, 0.9], [2.5, 0.5]]]
    assert client.put("/api/court/video/game.mp4", json=body).status_code == 422


def test_router_needs_tracks_to_find_guides(client, tmp_path):
    with patch.object(routes, "tracks_path", return_value=tmp_path / "none.jsonl"):
        assert client.get("/api/court/guides/game.mp4").status_code == 409


def test_router_rejects_unknown_landmarks_and_out_of_frame_marks(client):
    assert client.put("/api/court/video/game.mp4", json={"points": {"net_top": [0.5, 0.5]}}).status_code == 422
    assert client.put("/api/court/video/game.mp4", json={"points": {"left_near": [2.5, 0.5]}}).status_code == 422
    # A corner the camera cut off is still markable, a little past the edge.
    assert client.put(
        "/api/court/video/game.mp4", json={"points": {"left_near": [-0.2, 1.1]}, "frame_size": list(SIZE)}
    ).status_code == 200
    # The camera solve needs the frame the marks were taken in.
    assert client.put("/api/court/video/game.mp4", json={"points": {}}).status_code == 422


# ── Camera (3D) ──────────────────────────────────────────────────────────



def _pinhole(focal=1.2, center=(9.0, -12.0, 3.0), target=(9.0, 4.5, 1.0)):
    """A sideline camera as a court-metres → normalized-image function."""
    aspect = SIZE[0] / SIZE[1]
    c, t = np.array(center), np.array(target)
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    rot = np.vstack([x, np.cross(z, x), z])
    k = np.array([[focal, 0, aspect / 2], [0, focal, 0.5], [0, 0, 1]])

    def see(p):
        u = k @ (rot @ np.asarray(p, float) - rot @ c)
        return (float(u[0] / u[2] / aspect), float(u[1] / u[2]))

    return see


def _calibration(see, net_height=2.43):
    points = {n: see((*xy, 0.0)) for n, xy in geometry.LANDMARKS.items()}
    points |= {n: see((*xy, net_height)) for n, xy in geometry.NET_LANDMARKS.items()}
    return Calibration(points=points, net_height_m=net_height, frame_size=SIZE)


def test_camera_solve_recovers_focal_and_position():
    cam = camera.solve(_calibration(_pinhole()))
    assert cam.focal == pytest.approx(1.2, rel=1e-4)
    assert cam.center == pytest.approx((9.0, -12.0, 3.0), abs=1e-3)
    assert cam.off_floor == 2
    see = _pinhole()
    assert camera.project(cam, np.array([[4.0, 2.0, 3.0]]))[0] == pytest.approx(see((4.0, 2.0, 3.0)), abs=1e-6)


def test_camera_needs_floor_points():
    with pytest.raises(camera.CameraError, match="floor points"):
        camera.solve(Calibration(points={}, frame_size=SIZE))


def test_lift_recovers_contact_height_above_the_feet():
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    assert camera.lift(cam, see((4.0, 2.0, 3.1)), (4.0, 2.0)) == pytest.approx([4.0, 2.0, 3.1], abs=1e-3)


def test_lift_refuses_feet_behind_the_camera():
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    # A camera at y = -12: feet at y = -20 stand behind it.
    assert camera.lift(cam, see((4.0, 2.0, 3.1)), (4.0, -20.0)) is None


def test_grounded_feet_are_the_lowest_around_the_contact():
    tracklet = {
        "frames": [8, 10, 12, 14, 16],
        # Standing, feet low at takeoff, rising, airborne at the contact, landed.
        "boxes": [[0, 0, 10, 500], [0, 0, 10, 530], [0, 0, 10, 520], [0, 0, 10, 480], [0, 0, 10, 510]],
    }
    assert extraction_feet._grounded_box(tracklet, 14, window=4) == [0, 0, 10, 530]
    # Lost mid-jump: the landing still places them.
    assert extraction_feet._grounded_box(tracklet, 14, window=2) == [0, 0, 10, 520]
    assert extraction_feet._grounded_box(tracklet, 30, window=2) is None


def test_positions_outside_the_free_zone_or_reach_are_dropped():
    assert positions._in_play_area(np.array([-2.9, 11.9]))
    assert not positions._in_play_area(np.array([9.0, 15.4]))
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    assert positions._lift(cam, see((4.0, 2.0, 3.1)), np.array([4.0, 2.0])) is not None
    assert positions._lift(cam, see((4.0, 2.0, 6.0)), np.array([4.0, 2.0])) is None
    assert positions._lift(None, see((4.0, 2.0, 3.1)), np.array([4.0, 2.0])) is None


def test_arc_is_ballistic_between_its_ends():
    path = camera.arc(np.array([0.0, 0.0, 1.0]), np.array([10.0, 3.0, 1.0]), 1.0, samples=5)
    assert path[0] == pytest.approx([0, 0, 1]) and path[-1] == pytest.approx([10, 3, 1])
    assert path[2][2] == pytest.approx(1 + 9.81 / 8)  # apex at the midpoint: g·T²/8 above


def test_flights_never_jump_over_an_unplaced_touch():
    def touch(frame, rally=1):
        return {"id": f"e{frame}", "frame": frame, "time": frame / 30, "rally_id": rally,
                "label": "set", "ball_3d": np.array([frame / 10, 4.0, 2.0])}

    # Frame 45 was touched too but could not be placed (no actor).
    unplaced = {**touch(45), "ball_3d": None}
    events = [touch(0), touch(30), unplaced, touch(60), touch(90, rally=2)]
    flights = positions._flights(events, None, SIZE[0] / SIZE[1], None)
    assert [(f["from"], f["to"]) for f in flights] == [("e0", "e30")]
    assert flights[0]["source"] == "touches"


def test_compute_places_feet_lifts_contacts_and_keeps_reasons():
    see = _pinhole()
    calibration = _calibration(see)
    events = [
        # A set at (4, 2) touched at 2.5 m, then a spike there at 3.1 m.
        {"id": "f30", "frame": 30, "time": 1.0, "label": "set", "ball": see((4.0, 2.0, 2.5)), "rally_id": 1},
        {"id": "f60", "frame": 60, "time": 2.0, "label": "spike", "ball": see((4.0, 2.0, 3.1)), "rally_id": 1},
        {"id": "f75", "frame": 75, "time": 2.5, "label": "receive", "ball": None, "rally_id": 1},
        {"id": "f90", "frame": 90, "time": 3.0, "label": "score", "ball": see((12.0, 5.0, 0.0)), "rally_id": 1},
    ]
    feet = {"f30": see((4.0, 2.0, 0.0)), "f60": see((4.0, 2.0, 0.0)), "f75": "occluded"}
    result = positions.compute(calibration, events, feet, None)
    by_id = {e["id"]: e for e in result["events"]}
    assert by_id["f30"]["court_xy"] == pytest.approx([4.0, 2.0], abs=0.01)
    assert by_id["f60"]["ball_3d"][2] == pytest.approx(3.1, abs=0.01)
    assert by_id["f75"]["court_xy"] is None and by_id["f75"]["reason"] == "occluded"
    assert by_id["f90"]["court_xy"] == pytest.approx([12.0, 5.0], abs=0.01)
    assert [(a["from"], a["to"]) for a in result["arcs"]] == [("f30", "f60")]


def test_segments_place_each_touch_through_the_camera_of_its_time():
    # The camera moved at 2.5 s: a touch after it is seen through the new one.
    before, after = _pinhole(), _pinhole(center=(7.0, -11.0, 2.5))
    segments = positions.Segments([
        positions.Segment(start=0, calibration=_calibration(before)),
        positions.Segment(start=2.5, calibration=_calibration(after)),
    ])
    events = [
        {"id": "f30", "frame": 30, "time": 1.0, "label": "set", "ball": before((4.0, 2.0, 2.5)), "rally_id": 1},
        {"id": "f60", "frame": 60, "time": 2.0, "label": "spike", "ball": before((4.0, 2.0, 3.1)), "rally_id": 1},
        {"id": "f90", "frame": 90, "time": 3.0, "label": "set", "ball": after((6.0, 3.0, 2.5)), "rally_id": 1},
    ]
    feet = {"f30": before((4.0, 2.0, 0.0)), "f60": before((4.0, 2.0, 0.0)), "f90": after((6.0, 3.0, 0.0))}
    result = positions.compute_segments(segments, events, feet, None)
    by_id = {e["id"]: e for e in result["events"]}
    assert by_id["f60"]["court_xy"] == pytest.approx([4.0, 2.0], abs=0.01)
    assert by_id["f90"]["court_xy"] == pytest.approx([6.0, 3.0], abs=0.01)
    # No flight joins touches seen through two cameras.
    assert [(a["from"], a["to"]) for a in result["arcs"]] == [("f30", "f60")]


@pytest.mark.parametrize("starts", [[], [1.0], [0, 3.0, 3.0]])
def test_segments_cover_the_video_in_order(starts):
    calibration = _calibration(_pinhole())
    with pytest.raises(ValueError):
        positions.Segments.model_validate([{"start": s, "calibration": calibration} for s in starts])


def test_calibration_summary_projects_the_lines_back_onto_the_marks():
    see = _pinhole()
    summary = positions.calibration_summary(_calibration(see))
    assert summary["floor_rmse_m"] == pytest.approx(0.0, abs=1e-6)
    # The near sideline runs from the left-near corner to the right-near one.
    assert summary["lines"][0][0] == pytest.approx(see((0.0, 0.0, 0.0)), abs=1e-3)
    assert summary["lines"][0][1] == pytest.approx(see((18.0, 0.0, 0.0)), abs=1e-3)
    assert summary["net"][0] == pytest.approx(see((9.0, 0.0, 2.43)), abs=1e-3)
    assert summary["camera"]["center"] == pytest.approx([9.0, -12.0, 3.0], abs=0.01)
    # The projection lays any court point where the video shows it.
    p = np.array(summary["camera"]["projection"])
    h = p @ np.array([6.0, 4.0, 3.5, 1.0])
    assert h[:2] / h[2] == pytest.approx(see((6.0, 4.0, 3.5)), abs=1e-4)


def _flight(t):
    """A set from (4, 2, 2.5) towards (12, 6, 1): p(t) = p0 + v0·t + ½gt²."""
    p0, v0 = np.array([4.0, 2.0, 2.5]), np.array([8.0, 4.0, 3.4])
    t = np.asarray(t, float)[:, None]
    return p0 + v0 * t + 0.5 * np.array([0, 0, -9.81]) * t**2


def test_trajectory_fit_recovers_the_flight_among_clutter():
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    rng = np.random.default_rng(1)
    times = np.arange(0, 1.0, 1 / 15)
    ball = [see(p) for p in _flight(times)]
    # Every frame also boxes a head and a lamp, and two frames miss the ball.
    t = np.concatenate([np.delete(times, [3, 8]), times, times])
    xy = np.vstack([
        np.delete(np.array(ball), [3, 8], axis=0),
        rng.uniform(0.2, 0.8, (len(times), 2)),
        np.tile([0.9, 0.1], (len(times), 1)),
    ])
    fit = trajectory.fit(cam, SIZE[0] / SIZE[1], t, xy, 1.0, rng)
    assert fit is not None and fit.frames == len(times) - 2
    assert fit.at(np.array([0.0, 0.5, 1.0])) == pytest.approx(_flight([0.0, 0.5, 1.0]), abs=0.01)


def test_trajectory_fit_needs_frames_spread_over_the_flight():
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    rng = np.random.default_rng(1)
    early = np.arange(0, 0.4, 1 / 15)  # seen only in the first 40%
    xy = np.array([see(p) for p in _flight(early)])
    assert trajectory.fit(cam, SIZE[0] / SIZE[1], early, xy, 1.0, rng) is None
    few = early[:4]
    assert trajectory.fit(cam, SIZE[0] / SIZE[1], few, xy[:4], 0.2, rng) is None


def test_compute_fits_arcs_to_the_ball_track():
    see = _pinhole()
    calibration = _calibration(see)
    # The set is touched at frame 30 (1.0 s), the next touch 1 s later, no
    # feet for either: only the ball track can place this flight.
    events = [
        {"id": "f30", "frame": 30, "time": 1.0, "label": "set", "ball": None, "rally_id": 1},
        {"id": "f60", "frame": 60, "time": 2.0, "label": "spike", "ball": None, "rally_id": 1},
    ]
    frames = range(30, 61, 2)
    track = {"fps": 30.0, "frames": {
        str(f): [[*see(p), 0.9]] for f, p in zip(frames, _flight([(f - 30) / 30 for f in frames]))
    }}
    result = positions.compute(calibration, events, {}, track)
    (arc,) = result["arcs"]
    assert arc["source"] == "ball" and arc["frames"] == len(frames)
    assert arc["points"][0] == pytest.approx([4.0, 2.0, 2.5], abs=0.02)
    assert positions.compute(calibration, events, {}, None)["arcs"] == []


def test_open_fit_follows_a_ball_that_left_the_shot_down_to_the_floor():
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    rng = np.random.default_rng(1)
    # Seen for its first 0.4 s only, over a lamp boxed the whole second: the
    # lamp would win a closed fit's vote, and the ball covers too little of it.
    times = np.arange(0, 0.4, 1 / 15)
    lamp = np.arange(0, 1.0, 1 / 15)
    t = np.concatenate([times, lamp])
    xy = np.vstack([[see(p) for p in _flight(times)], np.tile([0.9, 0.1], (len(lamp), 1))])
    start = _flight([0.0])[0] + [0.3, 0.0, 0.2]  # placed from the feet: off by a little
    assert trajectory.fit(cam, SIZE[0] / SIZE[1], t, xy, 1.0, rng, [(0.0, start)]) is None
    fit = trajectory.fit_open(cam, SIZE[0] / SIZE[1], t, xy, start, rng)
    assert fit is not None and fit.frames == len(times)
    # Where it came down, 0.8 s past the last box, to a quarter metre.
    true_landing = (3.4 + np.sqrt(3.4**2 + 2 * 9.81 * 2.5)) / 9.81
    assert fit.landing() == pytest.approx(true_landing, abs=0.02)
    assert fit.at(np.array([fit.landing()]))[0] == pytest.approx(_flight([true_landing])[0], abs=0.25)


def test_compute_follows_a_flight_whose_next_touch_is_unplaced():
    see = _pinhole()
    calibration = _calibration(see)
    # The set at frame 30 is placed from its actor's feet; the next touch,
    # a dig past the free zone, is not — so no two-ended arc joins them.
    events = [
        {"id": "f30", "frame": 30, "time": 1.0, "label": "set", "ball": see(_flight([0.0])[0]), "rally_id": 1},
        {"id": "f60", "frame": 60, "time": 2.0, "label": "dig", "ball": None, "rally_id": 1},
    ]
    feet = {"f30": see((4.0, 2.0, 0.0)), "f60": "off court"}
    frames = range(32, 46, 2)
    track = {"fps": 30.0, "frames": {
        str(f): [[*see(p), 0.9]] for f, p in zip(frames, _flight([(f - 30) / 30 for f in frames]))
    }}
    assert positions.compute(calibration, events, feet, None)["arcs"] == []
    (arc,) = positions.compute(calibration, events, feet, track)["arcs"]
    assert (arc["from"], arc["to"], arc["source"]) == ("f30", "f60", "ball")
    # It flew the whole second to the dig, still above the floor.
    assert arc["end"] == pytest.approx(2.0)
    assert arc["points"][-1] == pytest.approx(_flight([1.0])[0], abs=0.1)


def test_compute_follows_a_flight_nothing_followed_down_to_the_floor():
    see = _pinhole()
    calibration = _calibration(see)
    events = [{"id": "f30", "frame": 30, "time": 1.0, "label": "set", "ball": see(_flight([0.0])[0]), "rally_id": 1}]
    frames = range(32, 46, 2)
    track = {"fps": 30.0, "frames": {
        str(f): [[*see(p), 0.9]] for f, p in zip(frames, _flight([(f - 30) / 30 for f in frames]))
    }}
    (arc,) = positions.compute(calibration, events, {"f30": see((4.0, 2.0, 0.0))}, track)["arcs"]
    assert arc["to"] is None
    # Seen until the last box, extrapolated from there to the floor.
    assert arc["seen_until"] == pytest.approx(1.0 + 14 / 30, abs=1e-3)
    assert arc["points"][-1][2] == pytest.approx(0.0, abs=0.01)
    assert arc["end"] == pytest.approx(1.0 + (3.4 + np.sqrt(3.4**2 + 2 * 9.81 * 2.5)) / 9.81, abs=0.03)


def test_a_serve_is_placed_behind_the_end_line_its_receive_faces():
    see = _pinhole()
    calibration = _calibration(see)
    # The receive is at x = 14, so the serves came from the x = 0 end.
    events = [
        {"id": "s1", "frame": 30, "time": 1.0, "label": "serve", "ball": see((-0.5, 3.0, 2.8)), "rally_id": 1},
        {"id": "r1", "frame": 60, "time": 2.0, "label": "receive", "ball": see((14.0, 4.0, 0.8)), "rally_id": 1},
        {"id": "s2", "frame": 300, "time": 10.0, "label": "serve", "ball": see((0.0, 6.0, 2.6)), "rally_id": 2},
        {"id": "r2", "frame": 330, "time": 11.0, "label": "receive", "ball": see((14.0, 4.0, 0.8)), "rally_id": 2},
        {"id": "s3", "frame": 600, "time": 20.0, "label": "serve", "ball": see((0.0, 6.0, 2.6)), "rally_id": 3},
    ]
    feet = {
        "s1": see((-0.5, 3.0, 0.0)),  # the server, behind the line: kept
        "r1": see((14.0, 4.0, 0.0)),
        "s2": see((13.0, 5.0, 0.0)),  # a receiver picked instead: the ball crosses the end line
        "r2": see((14.0, 4.0, 0.0)),
        "s3": see((13.0, 5.0, 0.0)),  # no receive to name the end: left alone
    }
    by_id = {e["id"]: e for e in positions.compute(calibration, events, feet, None)["events"]}
    assert by_id["s1"]["court_xy"] == pytest.approx([-0.5, 3.0], abs=0.01)
    assert by_id["s1"]["foot_image"] is not None
    assert by_id["s2"]["court_xy"] == pytest.approx([0.0, 6.0], abs=0.01)
    assert by_id["s2"]["ball_3d"] == pytest.approx([0.0, 6.0, 2.6], abs=0.01)
    assert by_id["s2"]["foot_image"] is None and by_id["s2"]["reason"] is None
    assert by_id["s3"]["court_xy"] == pytest.approx([13.0, 5.0], abs=0.01)
    # Without its ball the wrong actor is dropped, not kept.
    events[2] = {**events[2], "ball": None}
    by_id = {e["id"]: e for e in positions.compute(calibration, events, feet, None)["events"]}
    assert by_id["s2"]["court_xy"] is None and by_id["s2"]["reason"] == "not the server"


def test_ball_is_detected_up_to_each_next_touch_or_two_seconds_on():
    from yp_video.ball.track import flight_spans

    # At 30 fps: 10→40 and 40→70 chain; 200 has nothing within 60 frames
    # after it (330 is 130 on), so it gets 60; 330 is last.
    assert flight_spans([40, 10, 70, 200, 330], 30.0) == [(10, 130), (200, 260), (330, 390)]


def test_chain_meets_at_every_touch_and_finds_its_time():
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    rng = np.random.default_rng(1)
    # A set touched at (4, 2, 2.5) flies 0.7 s to a spike there, which drives
    # the ball on for 0.4 s. The spike was spotted 2 frames late.
    p1, v1 = _flight([0.7])[0], np.array([10.0, 2.0, -1.0])

    def spike(t):
        t = np.asarray(t, float)[:, None]
        return p1 + v1 * t + 0.5 * np.array([0, 0, -9.81]) * t**2

    late = 2 / 30
    times = [np.arange(0, 0.7, 1 / 30), np.arange(0, 0.4, 1 / 30)]
    flights = [
        trajectory.Flight(times[0], np.array([see(p) for p in _flight(times[0])]), 0.7 + late),
        trajectory.Flight(times[1] - late, np.array([see(p) for p in spike(times[1])]), 0.4 - late),
    ]
    chain = trajectory.fit_chain(cam, SIZE[0] / SIZE[1], flights, {}, rng)
    assert chain.touches[1] == pytest.approx(p1, abs=0.05)
    assert chain.shifts[1] == pytest.approx(-late, abs=0.01)
    assert [n for n, _ in chain.seen] == [len(times[0]), len(times[1])]
