"""How far the court calibrations can be trusted, on real footage.

  uv run python scripts/court_accuracy.py [stem ...]   # default: every calibration

Per calibrated video:
- floor LOO: drop one floor mark, fit the homography from the rest, and
  measure where the dropped mark lands on the court (metres) — what a
  floor position (feet, a landing) is worth;
- net tops: solve the camera from floor marks only and reproject the net
  tops — the off-plane check heights rest on (frame heights, and roughly
  metres at the net);
- contact heights: the lifted ball height per action label, against the
  range a player plausibly touches it in.

Gate for shipping 3D in the app (plan, 2026-10-03): median floor error under
0.3 m and at least 80% of contact heights plausible.
"""

from __future__ import annotations

import json
import statistics
import sys

import numpy as np

from yp_video.court import annotations, camera, geometry
from yp_video.web import court_positions

#: Contact heights a touch of each kind plausibly happens at (metres).
PLAUSIBLE_Z = {
    "receive": (0.2, 1.5),
    "set": (1.5, 3.4),
    "spike": (2.3, 3.8),
    "block": (2.2, 3.6),
    "serve": (1.8, 3.8),
}


def floor_loo(cal: annotations.Calibration) -> list[float]:
    floor = {n: xy for n, xy in cal.points.items() if n in geometry.LANDMARKS}
    if len(floor) <= geometry.MIN_POINTS:
        return []
    errors = []
    for name, xy in floor.items():
        rest = {n: p for n, p in floor.items() if n != name}
        try:
            fit = geometry.fit(rest)
        except geometry.FitError:
            continue
        landed = geometry.project(np.array(fit.image_to_court), np.array([xy]))[0]
        errors.append(float(np.linalg.norm(landed - np.array(geometry.LANDMARKS[name]))))
    return errors


def net_check(cal: annotations.Calibration) -> dict | None:
    nets = {n: xy for n, xy in cal.points.items() if n in geometry.NET_LANDMARKS}
    if not nets:
        return None
    floor_only = cal.model_copy(update={"points": {n: xy for n, xy in cal.points.items() if n in geometry.LANDMARKS}})
    try:
        cam = camera.solve(floor_only)
    except camera.CameraError:
        return None
    width, height = cal.frame_size
    aspect = width / height
    errors = []
    for name, xy in nets.items():
        x, y = geometry.NET_LANDMARKS[name]
        projected = camera.project(cam, np.array([[x, y, cal.net_height_m]]))[0]
        errors.append(float(np.linalg.norm((projected - np.array(xy)) * np.array([aspect, 1.0]))))
    full = camera.solve(cal)
    return {
        "net_error_frame_h": max(errors),
        "focal_floor_only": cam.focal,
        "focal_full": full.focal,
        "camera_center_m": [round(v, 2) for v in full.center],
        "rmse_full_frame_h": full.rmse,
    }


def heights(stem: str) -> dict[str, list[float]]:
    try:
        result = court_positions.compute(stem)
    except court_positions.NotReady as exc:
        print(f"  heights: skipped ({exc})")
        return {}
    by_label: dict[str, list[float]] = {}
    for event in result["events"]:
        if event.get("ball_3d") and event["label"] in PLAUSIBLE_Z:
            by_label.setdefault(event["label"], []).append(float(event["ball_3d"][2]))
    return by_label


def main() -> None:
    stems = sys.argv[1:] or sorted(
        p.name.removesuffix("_court.json") for p in annotations.annotation_path("x").parent.glob("*_court.json")
    )
    all_floor, plausible, placed = [], 0, 0
    report = {}
    for stem in stems:
        cal = annotations.load(stem)
        if cal is None:
            print(f"{stem}: no calibration")
            continue
        print(f"== {stem} ({len(cal.points)} marks, net {cal.net_height_m} m)")
        loo = floor_loo(cal)
        all_floor += loo
        if loo:
            print(f"  floor LOO: median {statistics.median(loo):.2f} m, max {max(loo):.2f} m over {len(loo)} marks")
        net = net_check(cal)
        if net:
            print(f"  net tops: {net['net_error_frame_h']:.4f} frame-h off; focal {net['focal_floor_only']:.2f} (floor) "
                  f"vs {net['focal_full']:.2f} (all); camera at {net['camera_center_m']}")
        z = heights(stem)
        for label, values in sorted(z.items()):
            lo, hi = PLAUSIBLE_Z[label]
            ok = sum(lo <= v <= hi for v in values)
            plausible += ok
            placed += len(values)
            print(f"  {label:8s} n={len(values):3d} median z {statistics.median(values):.2f} m, "
                  f"plausible {ok}/{len(values)}")
        report[stem] = {"floor_loo_m": loo, "net": net, "heights": z}
    print("== overall")
    if all_floor:
        print(f"  floor LOO median {statistics.median(all_floor):.2f} m over {len(all_floor)} marks "
              f"({'PASS' if statistics.median(all_floor) < 0.3 else 'FAIL'} < 0.3 m)")
    if placed:
        print(f"  plausible contact heights {plausible}/{placed} = {plausible / placed:.0%} "
              f"({'PASS' if plausible / placed >= 0.8 else 'FAIL'} ≥ 80%)")
    print(json.dumps({"calibrations": len(report)}))


if __name__ == "__main__":
    main()
