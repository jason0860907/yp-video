"""Actual ByteTrack and downstream records consume fusion boxes, without RF-DETR."""

from contextlib import nullcontext
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from yp_video.action import predict
from yp_video.action.spot_pass import RallyOptions, SpotOptions, SpotPassResult
from yp_video.core import person_boxes as boxes
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.extraction import pipeline
from yp_video.extraction import store as extraction_store
from yp_video.tracklets import fusion
from yp_video.tracklets import store as tracks_store
from yp_video.web import fusion_inference as fi


def write_boxes(path, *, stride=1, num_frames=90, empty=(), checkpoint="test"):
    frames = np.arange(0, num_frames, stride, dtype=np.int32)
    counts = np.array([0 if f in empty else 1 for f in frames], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, frames=frames, counts=counts,
        boxes=np.tile([0.1, 0.2, 0.3, 0.8], (int(counts.sum()), 1)).astype(np.float32),
        scores=np.full(int(counts.sum()), 0.9, dtype=np.float32),
        stride=stride, num_frames=num_frames, checkpoint=checkpoint,
    )


@pytest.fixture
def layout(tmp_path, monkeypatch):
    monkeypatch.setattr(boxes, "TRACKS_DIR", tmp_path / "tracks")
    monkeypatch.setattr(tracks_store, "TRACKS_DIR", tmp_path / "tracks")
    monkeypatch.setattr(extraction_store, "RECORDS_DIR", tmp_path / "records")
    monkeypatch.setattr(pipeline, "RECORDS_DIR", tmp_path / "records")
    monkeypatch.setattr(fusion, "rally_fingerprint", lambda stem: "rallies-v1")
    monkeypatch.setattr(tracks_store, "rally_fingerprint", lambda stem: "rallies-v1")
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 1000, cv2.CAP_PROP_FRAME_HEIGHT: 500,
        cv2.CAP_PROP_FPS: 30.0,
    }[prop]
    cap.read.side_effect = AssertionError("Fusion tracking/detection must not decode images")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: cap)
    monkeypatch.setattr(pipeline, "person_detector", Mock(side_effect=AssertionError("No RF-DETR")))
    return tmp_path


def test_bytetrack_keeps_native_frames_resets_per_rally_and_removes_old_masks(layout, monkeypatch):
    video = layout / "match.mp4"
    path = boxes.person_boxes_path(video.stem)
    write_boxes(path, stride=2, empty=(12, 14))
    monkeypatch.setattr(fusion, "load_rallies", lambda stem: [
        {"rally_id": 3, "start": 0, "end": 1},
        {"rally_id": 7, "start": 2, "end": 2.9},
    ])
    masks = tracks_store.tracks_masks_path(video.stem)
    masks.touch()
    counts = fusion.track_person_boxes(video)
    header, records = read_jsonl(tracks_store.tracks_path(video.stem))
    assert counts == {"rallies": 2, "frames": 30, "tracklets": 2}
    assert [(r["rally_id"], r["track_id"]) for r in records] == [(3, 1), (7, 1)]
    assert records[0]["boxes"][0] == [100, 100, 300, 400]
    assert all(f % 2 == 0 for r in records for f in r["frames"])
    assert 12 not in records[0]["frames"] and 16 in records[0]["frames"]
    assert header["stride"] == 2 and "mask_res" not in header
    assert not masks.exists()
    assert fusion.fusion_tracks_current(video.stem)
    write_boxes(path, stride=2)
    assert not fusion.fusion_tracks_current(video.stem)


def test_detection_uses_fusion_frames_outside_rallies_and_preserves_picks(layout, monkeypatch):
    video = layout / "match.mp4"
    path = boxes.person_boxes_path(video.stem)
    write_boxes(path, stride=2, empty=(12,))
    events = [{"frame": 11, "label": "serve"}, {"frame": 12, "label": "set"}]
    monkeypatch.setattr(pipeline, "load_events", lambda stem: events)
    picked = {"frame": 11, "id": "f11", "status": "ok", "box": [1, 2, 3, 4]}
    write_jsonl(extraction_store.records_path(video.stem), {}, [picked])
    result = pipeline.detect_video(video, person_boxes=path)
    meta, records = read_jsonl(extraction_store.records_path(video.stem))
    assert result == {"events": 2, "detections": 1, "undecodable": 0}
    assert records[0]["detections"][0]["box"] == [100, 100, 300, 400]
    assert records[0]["box"] == picked["box"]
    assert records[1]["detections"] == []
    assert meta["source"]["detector"] == boxes.DETECTOR_NAME
    assert meta["source"]["stride"] == 2
    assert pipeline.detections_current(video.stem, detector=boxes.DETECTOR_NAME)
    write_boxes(path, stride=2)
    assert not pipeline.detections_current(video.stem, detector=boxes.DETECTOR_NAME)


def test_incomplete_archive_is_not_treated_as_empty_detections(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez(path, frames=[0, 2], counts=[0, 0], boxes=np.empty((0, 4)), scores=[], stride=1, num_frames=3)
    with pytest.raises(ValueError, match="every sampled frame"):
        boxes.PersonBoxes.load(path)


def test_event_outside_archive_is_an_error(tmp_path):
    path = tmp_path / "people.npz"
    write_boxes(path, stride=2, num_frames=10)
    people = boxes.PersonBoxes.load(path)
    assert len(people.for_frame(9, 1000, 500)) == 1
    with pytest.raises(ValueError, match="outside person output"):
        people.for_frame(10, 1000, 500)


def test_subprocess_person_output_survives_temp_dir_and_missing_head_fails(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "source.npz"
    write_boxes(source)
    # Source subprocess output does not yet contain the checkpoint provenance.
    with np.load(source) as data:
        raw = {key: data[key] for key in data.files if key != "checkpoint"}
    np.savez(source, **raw)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    target = tmp_path / "retained.npz"
    monkeypatch.setattr(predict.prelabel, "spot_available", lambda: True)

    def command(**kwargs):
        dest = kwargs["save_dir"]
        return [sys.executable, "-c", (
            "from pathlib import Path; import shutil; "
            f"p=Path({str(dest)!r}); (p/'action').mkdir(); "
            "(p/'action'/'predictions.json').write_text('[]'); "
            "(p/'person').mkdir(); "
            f"shutil.copyfile({str(source)!r}, p/'person'/'boxes.npz')"
        )]

    monkeypatch.setattr(predict.prelabel, "build_command", command)
    predict.run_spot_inference("video.mp4", checkpoint=checkpoint, tasks=("action",), person_output=target)
    assert len(boxes.PersonBoxes.load(target).frames) == 90
    with np.load(target) as data:
        assert str(data["checkpoint"]) == boxes.checkpoint_identity(checkpoint)
    # A checkpoint with no person output must fail, never silently retain old boxes.
    monkeypatch.setattr(predict.prelabel, "build_command", lambda **kw: [sys.executable, "-c", (
        "from pathlib import Path; "
        f"p=Path({str(kw['save_dir'])!r})/'action'; p.mkdir(); "
        "(p/'predictions.json').write_text('[]')"
    )])
    with pytest.raises(predict.SpotInferenceError, match="person head"):
        predict.run_spot_inference("video.mp4", checkpoint=checkpoint, tasks=("action",), person_output=target)


def test_full_pipeline_uses_one_spot_pass_and_new_boxes_even_when_labels_exist(layout, monkeypatch):
    video = layout / "match.mp4"
    checkpoint = layout / "checkpoint.pt"
    checkpoint.touch()
    rallies = [{"rally_id": 1, "start": 0, "end": 2.9}]
    events = [{"frame": 20, "label": "spike"}]
    monkeypatch.setattr(fi, "materialized_cut", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(fi, "plan_spot_stages", lambda *args, **kwargs: {
        "rally": "kept existing rallies", "action": "kept existing actions",
    })
    monkeypatch.setattr(fi, "load_rallies", lambda stem: rallies)
    monkeypatch.setattr(fusion, "load_rallies", lambda stem: rallies)
    monkeypatch.setattr(fi, "load_events", lambda stem: events)
    monkeypatch.setattr(pipeline, "load_events", lambda stem: events)

    def spot(source, **kwargs):
        assert kwargs["tasks"] == ("rally", "action")
        assert kwargs.get("rallies") is None
        write_boxes(kwargs["person_output"], checkpoint=boxes.checkpoint_identity(checkpoint))
        return SpotPassResult(30, 90, 3, checkpoint, 0.15, rallies, [])

    run_spot = Mock(side_effect=spot)
    monkeypatch.setattr(fi, "run_spot_pass", run_spot)

    def associate(**kwargs):
        _, records = read_jsonl(extraction_store.records_path(video.stem))
        assert records[0]["detections"]
        assert fusion.fusion_tracks_current(video.stem)
        return {"changed": 1}

    monkeypatch.setattr(fi, "run_association_stage", associate)
    options = dict(
        video=video, checkpoint=checkpoint, clip_checkpoint=checkpoint,
        rally=RallyOptions(0.5, 2, 4), action_min_score=0.15,
        spot=SpotOptions(4, 0, 64), overwrite=False, on_progress=lambda *args: None,
    )
    result = fi.run_video(**options)
    assert result.tracklets == 1 and result.detections == 1
    assert result.rallies is None and result.events is None
    assert result.association == {"changed": 1}
    fi.run_video(**options)
    assert run_spot.call_count == 1  # Retained boxes/tracks are reused on the second run.
