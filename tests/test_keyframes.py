from __future__ import annotations


from nfl_gsplat.calibration.keyframes import (
    Keyframe, load_keyframes, save_keyframes,
)


def test_keyframes_roundtrip(tmp_path):
    kfs = [
        Keyframe(frame=0, landmarks={"mid_50_left_hash": (1199.0, 654.0)}),
        Keyframe(frame=300, landmarks={"home_25_right_hash": (1101.0, 459.0)}),
    ]
    p = tmp_path / "sideline_keyframes.json"
    save_keyframes(p, kfs)
    got = load_keyframes(p)
    assert [k.frame for k in got] == [0, 300]
    assert got[0].landmarks["mid_50_left_hash"] == (1199.0, 654.0)


def test_load_keyframes_sorted_by_frame(tmp_path):
    p = tmp_path / "kf.json"
    save_keyframes(p, [Keyframe(5, {"a": (1.0, 2.0)}), Keyframe(1, {"b": (3.0, 4.0)})])
    assert [k.frame for k in load_keyframes(p)] == [1, 5]
