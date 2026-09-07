"""endzone_only_rule: endzone-only ids leave the timeline; two-view and sideline ids stay."""
import pandas as pd

from nfl_gsplat.render.endzone_only_rule import endzone_only_ids


def _df(rows):
    return pd.DataFrame(rows, columns=["frame", "cam", "global_player_id"])


def test_endzone_only_ids_are_dropped_and_two_view_ids_kept():
    df = _df([
        (0, "sideline", 1), (1, "sideline", 1),                       # sideline-only: kept
        (0, "endzone", 2), (1, "endzone", 2),                         # endzone-only: dropped
        (0, "sideline", 3), (0, "endzone", 3), (1, "endzone", 3),     # two-view once: kept
        (0, "endzone", -1),                                           # unlinked: ignored
    ])
    views = {0: {1: ["sideline"], 2: ["endzone"], 3: ["sideline", "endzone"]},
             1: {1: ["sideline"], 2: ["endzone"], 3: ["endzone"]}}
    assert endzone_only_ids(df, views) == {2}


def test_the_camera_name_is_a_parameter():
    df = _df([(0, "sideline", 5), (0, "endzone", 6)])
    views = {0: {5: ["sideline"], 6: ["endzone"]}}
    assert endzone_only_ids(df, views, cam="sideline") == {5}


def test_frustum_aware_keeps_an_endzone_only_id_the_sideline_cannot_see():
    import numpy as np

    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 0.0]))
    n = 4
    track = CameraTrack(K=np.stack([K] * n), R=np.stack([R] * n), t=np.stack([t] * n), conf=np.ones(n),
                        width=1920, height=1080)
    df = _df([(f, "endzone", 7) for f in range(n)] + [(f, "endzone", 8) for f in range(n)])
    views = {f: {7: ["endzone"], 8: ["endzone"]} for f in range(n)}
    ground = {f: {7: (0.0, 0.0), 8: (40.0, 0.0)} for f in range(n)}     # 7 at the aim point, 8 forty metres off-axis
    assert endzone_only_ids(df, views, ground=ground, sideline=track) == {7}
    assert endzone_only_ids(df, views) == {7, 8}                         # no cameras: the old behaviour
