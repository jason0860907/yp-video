"""Court calibration: the fit recovers a known camera, and the store and
router round-trip marks without keeping a derived matrix."""
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yp_video.court import annotations as store
from yp_video.court import camera, geometry
from yp_video.court.annotations import Calibration
from yp_video.web import court_positions
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


def test_router_rejects_unknown_landmarks_and_out_of_frame_marks(client):
    assert client.put("/api/court/video/game.mp4", json={"points": {"net_top": [0.5, 0.5]}}).status_code == 422
    assert client.put("/api/court/video/game.mp4", json={"points": {"left_near": [1.5, 0.5]}}).status_code == 422
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
    assert court_positions._grounded_box(tracklet, 14, window=4) == [0, 0, 10, 530]
    # Lost mid-jump: the landing still places them.
    assert court_positions._grounded_box(tracklet, 14, window=2) == [0, 0, 10, 520]
    assert court_positions._grounded_box(tracklet, 30, window=2) is None


def test_positions_outside_the_free_zone_or_reach_are_dropped():
    assert court_positions._in_play_area(np.array([-2.9, 11.9]))
    assert not court_positions._in_play_area(np.array([9.0, 15.4]))
    see = _pinhole()
    cam = camera.solve(_calibration(see))
    assert court_positions._lift(cam, see((4.0, 2.0, 3.1)), np.array([4.0, 2.0])) is not None
    assert court_positions._lift(cam, see((4.0, 2.0, 6.0)), np.array([4.0, 2.0])) is None
    assert court_positions._lift(None, see((4.0, 2.0, 3.1)), np.array([4.0, 2.0])) is None


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
    flights = court_positions._flights(events)
    assert [(f["from"], f["to"]) for f in flights] == [("e0", "e30")]
