import numpy as np

from nfl_gsplat.calibration.roboflow_kps import load_kps_json
from nfl_gsplat.calibration.roboflow_precompute import run_precompute


def _frames(n):
    img = np.zeros((10, 10, 3), np.uint8)
    for i in range(n):
        yield i, img


def test_precompute_writes_filtered_cache(tmp_path):
    def infer_fn(bgr):
        return [("30", 100.0, 50.0, 0.9),        # kept
                ("30-top-sl", 5.0, 5.0, 0.9),    # sideline -> dropped
                ("20", 10.0, 10.0, 0.2)]         # low conf -> dropped
    out = tmp_path / "roboflow_kps.json"
    n_hit = run_precompute(_frames(3), infer_fn=infer_fn, model_id="m/2",
                           video_name="v.mp4", num_frames=3, out_json=out,
                           kp_conf=0.5)
    assert n_hit == 3
    loaded = load_kps_json(out, expect_num_frames=3)
    assert loaded[0] == [("30", 100.0, 50.0, 0.9)]
    assert len(loaded) == 3


def test_precompute_stride_leaves_frames_absent(tmp_path):
    # caller controls stride by what frames_iter yields; absent frames stay absent
    def infer_fn(bgr):
        return [("30", 1.0, 2.0, 0.9)]
    out = tmp_path / "kps.json"
    frames = ((i, np.zeros((4, 4, 3), np.uint8)) for i in (0, 5))
    run_precompute(frames, infer_fn=infer_fn, model_id="m/2", video_name="v.mp4",
                   num_frames=10, out_json=out, kp_conf=0.5)
    loaded = load_kps_json(out, expect_num_frames=10)
    assert set(loaded) == {0, 5}
