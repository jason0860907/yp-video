"""Human review never silently becomes an empty or stale training target."""
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from yp_video.actor.person_labels import write_person_labels
from yp_video.person import annotations as store
from yp_video.web import detection_media, detection_predictions
from yp_video.web.routers import detection_label as routes

BOX = [0.1, 0.2, 0.3, 0.8]


@pytest.fixture
def labels(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'PERSON_ANNOTATIONS_DIR', tmp_path / 'labels')
    return tmp_path


def test_reviewed_empty_draft_unknown_and_conflict(labels):
    assert store.load('game') is None
    reviewed = store.save('game', 30, 1, store.FrameAnnotation(state='reviewed', boxes=[BOX]))
    store.save('game', 30, 2, store.FrameAnnotation(state='reviewed', boxes=[]))
    store.save('game', 30, 3, store.FrameAnnotation(boxes=[BOX]))
    pseudo = {1: [[0, 0, 1, 1]], 2: [BOX], 3: [BOX], 4: [BOX]}
    assert store.apply_annotations('game', 30, pseudo) == 2
    assert pseudo == {1: [tuple(BOX)], 2: [], 4: [BOX]}
    assert 5 not in pseudo  # Unlabeled is not an empty negative.
    with pytest.raises(store.RevisionConflict):
        store.save('game', 30, 1, store.FrameAnnotation(boxes=[]))
    assert store.load('game').frames[1] == reviewed
    store.save('game', 30, 1, store.FrameAnnotation(revision=1, state='draft', boxes=[BOX]))
    store.apply_annotations('game', 30, pseudo)
    assert 1 not in pseudo  # Editing a reviewed frame revokes supervision.


@pytest.mark.parametrize('box', [[0, 0, 0, 1], [0, 1, 1, 0], [-1, 0, 1, 1], [0, 0, float('nan'), 1]])
def test_invalid_boxes(box):
    with pytest.raises(ValidationError):
        store.FrameAnnotation(boxes=[box])


def test_no_tracks_still_exports_human_frames_and_empty(labels):
    store.save('game', 3, 0, store.FrameAnnotation(state='reviewed', boxes=[BOX]))
    store.save('game', 3, 1, store.FrameAnnotation(state='reviewed'))
    store.save('game', 3, 2, store.FrameAnnotation(boxes=[BOX]))
    cache = labels / 'cache' / 'game'
    cache.mkdir(parents=True)
    for frame in range(3):
        (cache / f'{frame:06d}.jpg').write_bytes(b'frame')
    with patch('yp_video.actor.person_labels.tracks_path', return_value=labels / 'missing.jsonl'):
        result = write_person_labels([(labels / 'absent.jsonl', labels / 'game.mp4')], label_dir=labels / 'out', cache_root=cache.parent)
    assert result['reviewed_frames'] == 2
    with np.load(labels / 'out' / 'game_person.npz') as output:
        assert output['frames'].tolist() == [0, 1]
        assert output['counts'].tolist() == [1, 0]
        np.testing.assert_allclose(output['boxes'], [BOX], atol=0.001)
    with pytest.raises(ValueError, match='frame count mismatch'):
        store.apply_annotations('game', 4, {})


def test_api_save_reload_drafts_and_prediction_separation(labels, monkeypatch):
    monkeypatch.setattr(routes, 'resolve_cut', lambda name: labels / name if name == 'game.mp4' else None)
    monkeypatch.setattr(detection_media, 'metadata', lambda video: {'num_frames': 4, 'fps': 30})
    monkeypatch.setattr(routes, 'load_events', lambda stem: [{'frame': 1}])
    monkeypatch.setattr(detection_predictions, 'person_boxes_path', lambda stem: labels / 'predictions.npz')
    monkeypatch.setattr(routes, 'sync_to_r2', lambda *a: None)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    assert client.get('/frame/game.mp4?frame=0').json()['annotation'] is None
    assert client.put('/frame/game.mp4?frame=4', json={'boxes': []}).status_code == 422
    assert client.put('/frame/game.mp4?frame=0', json={'state': 'reviewed', 'boxes': [BOX]}).status_code == 200
    assert client.put('/frame/game.mp4?frame=0', json={'boxes': []}).status_code == 409
    frame = client.get('/frame/game.mp4?frame=0').json()
    assert frame['annotation']['boxes'] == [BOX]
    assert frame['prediction']['boxes'] == []
    assert client.get('/video/game.mp4').json()['frames']['0']['state'] == 'reviewed'
    assert client.get('/frame/unknown.mp4?frame=0').status_code == 404
    np.savez(labels / 'predictions.npz', frames=[0, 2], counts=[1, 0], boxes=[BOX], scores=[0.9], stride=2, num_frames=4)
    frame = client.get('/frame/game.mp4?frame=1').json()
    assert frame['annotation'] is None
    assert frame['prediction']['frame'] == 0
    assert frame['prediction']['scores'] == [0.9]
    assert set(store.load('game').frames) == {0}


def test_native_cache_pts_time_base_and_bounds(labels, monkeypatch):
    directory = labels / 'frames' / 'game'
    directory.mkdir(parents=True)
    (directory / 'metadata.json').write_text(json.dumps({'sampling': 'native', 'frames': 3, 'time_base': [1, 30000]}))
    np.save(directory / 'pts.npy', [0, 1001, 2002])
    (directory / '000001.jpg').write_bytes(b'exact-frame-1')
    monkeypatch.setattr(detection_media, 'ACTION_FRAMES_DIR', directory.parent)
    video = labels / 'game.mp4'
    meta = detection_media.metadata(video)
    assert meta['num_frames'] == 3
    assert meta['fps'] == pytest.approx(30000 / 1001)
    assert detection_media.frame_image(video, 1) == b'exact-frame-1'


def test_existing_tracking_and_event_detections_are_named_sources(labels, monkeypatch):
    track = labels / 'tracks.jsonl'
    detections = labels / 'detections.jsonl'
    meta = {'_meta': True, 'fps': 30, 'frame_size': [100, 100], 'source': {'detector': 'test-detector'}}
    track.write_text('\n'.join(json.dumps(row) for row in [meta, {'rally_id': 1, 'track_id': 1, 'frames': [10, 11], 'boxes': [[10, 20, 30, 80]] * 2, 'scores': [.9, .8]}]))
    detections.write_text('\n'.join(json.dumps(row) for row in [meta, {'frame': 15, 'detections': [{'box': [20, 10, 50, 90], 'score': .7}]}]))
    monkeypatch.setattr(detection_predictions, 'tracks_path', lambda stem: track)
    monkeypatch.setattr(detection_predictions, 'records_path', lambda stem: detections)
    monkeypatch.setattr(detection_predictions, 'person_boxes_path', lambda stem: labels / 'absent.npz')
    sources = detection_predictions.sources('existing', 30, 30)
    assert [s['id'] for s in sources] == ['tracking', 'detection']
    assert sources[0]['first_frame'] == 10
    tracking = detection_predictions.prediction('existing', 10, 30, 30, None)
    assert tracking['source'] == 'tracking'
    assert tracking['boxes'] == [BOX]
    assert tracking['next_frame'] == 11
    # No switching from the selected source to another source at a missing frame.
    gap = detection_predictions.prediction('existing', 15, 30, 30, 'tracking')
    assert gap['boxes'] == []
    assert gap['previous_frame'] == 11
    event = detection_predictions.prediction('existing', 15, 30, 30, 'detection')
    assert event['boxes'] == [[.2, .1, .5, .9]]
    assert detection_predictions.prediction('existing', 10, 30, 60, 'tracking')['boxes'] == []
