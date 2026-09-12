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


def test_beyond_sideline_span_drops_the_endzone_tail_the_sideline_could_see():
    import numpy as np
    import pandas as pd

    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    # id 1: sideline frames 10..20, endzone frames 0..60; id 2: sideline only, 0..60
    rows = [{"cam": "sideline", "track_id": 1, "frame": f} for f in range(10, 21)]
    rows += [{"cam": "endzone", "track_id": 1, "frame": f} for f in range(0, 61)]
    rows += [{"cam": "sideline", "track_id": 2, "frame": f} for f in range(0, 61)]
    df = pd.DataFrame(rows)
    ground = {f: {1: np.array([5.0, 0.0]), 2: np.array([-5.0, 0.0])} for f in range(0, 61)}
    # far downfield from frame 50 on: outside the sideline image
    for f in range(50, 61):
        ground[f][1] = np.array([80.0, 0.0])
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 0.0]))
    n = 61
    track = CameraTrack(K=np.stack([K] * n), R=np.stack([R] * n), t=np.stack([t] * n), conf=np.ones(n),
                        width=1920, height=1080)                     # sees the field around x = 0
    out, dropped = beyond_sideline_span(ground, df, track, gap=5)
    for f in range(0, 61):
        assert 2 in out[f]                                            # the sideline's own id: untouched
        if 5 <= f <= 25:
            assert 1 in out[f], f                                     # the span plus the gap
        elif f >= 50:
            assert 1 in out[f], f                                     # beyond the span, unseen: kept
        else:
            assert 1 not in out[f], f                                 # beyond the span, in view: dropped
    assert dropped == 5 + (50 - 26)


def test_beyond_the_span_is_kept_when_the_sideline_has_nobody_there():
    """Past its sideline span an id is a second copy only if the sideline draws that man."""
    import numpy as np
    import pandas as pd
    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    class _Side:
        # a camera at the origin looking down +y; everything in front of it is inside the image
        K = [np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])] * 400
        R = [np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])] * 400
        t = [np.zeros(3)] * 400
        conf = np.ones(400)
        width, height = 1920, 1080

    df = pd.DataFrame({"cam": ["sideline"] * 3, "frame": [10, 11, 12], "track_id": [1, 1, 1]})
    ground = {300: {1: np.array([0.5, 40.0])}}                  # id 1, long past its sideline span
    side = {300: {2: np.array([0.6, 40.1])}}                    # ... and the sideline draws that man as id 2
    out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side)
    assert dropped == 1 and out[300] == {}
    side_far = {300: {2: np.array([9.0, 40.0])}}                # nobody near: the sideline lost him
    out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side_far)
    assert dropped == 0 and 1 in out[300]
