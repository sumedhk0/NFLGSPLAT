def test_sample_frame_indices_even_spread():
    from nfl_gsplat.landmarks.labeling import sample_frame_indices
    idx = sample_frame_indices(num_frames=1000, count=10)
    assert len(idx) == 10 and idx[0] == 0 and idx[-1] == 999
    assert all(b > a for a, b in zip(idx, idx[1:]))


def test_build_label_record_shape():
    from nfl_gsplat.landmarks.labeling import build_label_record
    rec = build_label_record("f0.png", [("away_30_left_hash", (10.0, 20.0))])
    assert rec == {"file": "f0.png",
                   "points": [{"name": "away_30_left_hash", "uv": [10.0, 20.0]}]}
