from yp_video import config


def test_user_kinds_registered():
    assert config.USER_CUT_KINDS == {
        "baseline": "user-baseline",
        "corner": "user-corner",
        "high": "user-high",
        "center": "user-center",
    }
    for kind in config.USER_CUT_KINDS.values():
        info = config.CUT_KINDS[kind]
        assert info.r2_category == f"cuts-{kind}"
        assert info.local_dir == config.VIDEOS_DIR / f"cuts-{kind}"
        assert info.r2_category in config.R2_CATEGORIES
        assert info.local_dir in config.CUTS_DIRS
        assert config.cut_kind_of(info.local_dir / "x.mp4") == kind


def test_annotation_stats_counts_user_kinds(tmp_path, monkeypatch):
    from yp_video.action import training

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "m1_actions.jsonl").write_text(
        '{"_meta": true, "video": "m1", "num_frames": 10}\n'
        '{"frame": 1, "label": "serve"}\n'
    )
    monkeypatch.setattr(training, "ACTION_ANNOTATIONS_DIR", labels)
    monkeypatch.setattr(training, "read_val_set_file", lambda: [])
    video = config.CUT_KINDS["user-corner"].local_dir / "m1.mp4"
    stats = training.annotation_stats(lambda name: video)
    assert set(stats["by_view"]) == set(config.CUT_KINDS)
    assert stats["by_view"]["user-corner"]["videos"] == 1


def test_detector_contract_angles_are_the_user_angles():
    from yp_video.contracts.detector import CameraAngle

    assert tuple(a.value for a in CameraAngle) == config.USER_ANGLES
