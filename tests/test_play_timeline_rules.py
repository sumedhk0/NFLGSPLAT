"""render.edge_rule.edge_clipped_ids: one-view ids whose boxes all touch a frame edge."""
import pandas as pd

from nfl_gsplat.render.edge_rule import EDGE_PX, edge_clipped_ids


class _Track:
    height = 1080
    width = 1920


def _rows(pid, cam, boxes):
    return [{"frame": i, "cam": cam, "track_id": pid, "global_player_id": pid,
             "bbox_x1": b[0], "bbox_y1": b[1], "bbox_x2": b[2], "bbox_y2": b[3]}
            for i, b in enumerate(boxes)]


def test_top_clipped_endzone_only_id_is_dropped_two_view_and_interior_are_kept():
    rows = []
    rows += _rows(1, "endzone", [(100, 0, 140, 70), (120, 3, 160, 75)])          # clipped at the top
    rows += _rows(2, "endzone", [(100, 200, 140, 350), (120, 0, 160, 75)])       # one interior box: kept
    rows += _rows(3, "endzone", [(100, 0, 140, 70)]) + _rows(3, "sideline", [(500, 400, 560, 560)])
    rows += _rows(4, "sideline", [(500, 1000, 560, 1080 - 2)])                   # clipped at the bottom
    df = pd.DataFrame(rows)
    views = {0: {3: ("endzone", "sideline")}}
    tracks = {"endzone": _Track(), "sideline": _Track()}
    out = edge_clipped_ids(df, tracks, views)
    assert out == {1, 4}
    assert EDGE_PX >= 1.0


class _EndzoneTrack:
    """A camera on the ground at (-100, 0.5), 5 m up, looking +x (only its ground centre matters here)."""

    def __init__(self, n=40):
        import numpy as np
        R = np.eye(3)
        eye = np.array([-100.0, 0.5, 5.0])
        self.R = np.repeat(R[None], n, axis=0)
        self.t = np.repeat((-R @ eye)[None], n, axis=0)
        self.conf = np.ones(n)


def test_hold_blind_axis_slides_endzone_only_frames_along_the_endzone_ray():
    import numpy as np

    from nfl_gsplat.render.blind_axis import hold_blind_axis
    from nfl_gsplat.render.depth_snap import camera_ground_centre

    track = _EndzoneTrack()
    ground, views = {}, {}
    for f in range(10, 21):                      # the sideline sees him walking x 0 -> 2
        ground[f] = {1: np.array([0.2 * (f - 10), 1.0])}
        views[f] = {1: ("sideline", "endzone")}
    for f in range(21, 26):                      # endzone only: its foot point says x = 5 (wrong by 3 m)
        ground[f] = {1: np.array([5.0, 1.0])}
        views[f] = {1: ("endzone",)}
    ground[26] = {1: np.array([2.6, 1.0])}
    views[26] = {1: ("sideline",)}
    ground[60] = {1: np.array([9.0, 1.0])}       # far beyond the window: untouched
    views[60] = {1: ("endzone",)}
    out, moves = hold_blind_axis(ground, views, track, window=30)
    assert len(moves) == 5 and all(2.0 < m < 3.5 for m in moves)
    centre = camera_ground_centre(track, 23)
    for f in range(21, 26):
        q = out[f][1]
        w = (f - 20) / 6.0
        assert abs(q[0] - ((1 - w) * 2.0 + w * 2.6)) < 1e-9          # x interpolated between the sightings
        v0, v1 = ground[f][1] - centre, q - centre
        assert abs(np.cross(v0, v1)) < 1e-9 and v1 @ v0 > 0            # still on the same endzone ray
    assert np.allclose(out[60][1], [9.0, 1.0])
    for f in range(10, 21):
        assert np.allclose(out[f][1], ground[f][1])                   # seen frames never move
    # a run whose median slide is beyond max_move_m is refused whole; the input is never mutated
    out2, moves2 = hold_blind_axis(ground, views, track, window=30, max_move_m=1.0)
    assert moves2 == [] and np.allclose(out2[23][1], [5.0, 1.0]) and np.allclose(ground[23][1], [5.0, 1.0])


def test_hold_blind_axis_one_sided_hold_reaches_only_a_few_frames():
    """An endzone-only run with a sideline sighting on one side only is held from it within ``reach``
    frames and left alone beyond: pinning a moving man's x to a sighting a second old made an
    eight-frame fragment hop 0.4 m/frame on play 1."""
    import numpy as np

    from nfl_gsplat.render.blind_axis import hold_blind_axis

    track = _EndzoneTrack(n=120)
    ground, views = {}, {}
    for f in range(10, 21):
        ground[f] = {1: np.array([2.0, 1.0])}
        views[f] = {1: ("sideline",)}
    for f in range(21, 60):                      # a long endzone-only tail, x wandering
        ground[f] = {1: np.array([4.0 + 0.1 * (f - 21), 1.0])}
        views[f] = {1: ("endzone",)}
    out, moves = hold_blind_axis(ground, views, track, window=30, reach=6)
    # the tail is ONE run with no sighting after it and a sighting 1 frame before: within reach as a
    # run, so the whole run is held only if its median slide is under the cap -- here it is not
    # (median x error ~3.9 m > 4.0? no: 4.0 + 1.9 - 2.0 = 3.9 -> under), so the run is held whole
    assert len(moves) == 39 and all(abs(out[f][1][0] - 2.0) < 1e-9 for f in range(21, 60))
    # push the run out of reach: the sighting is 10 frames before it
    ground2 = {f: v for f, v in ground.items() if f < 11 or f >= 21}
    views2 = {f: v for f, v in views.items() if f < 11 or f >= 21}
    out2, moves2 = hold_blind_axis(ground2, views2, track, window=30, reach=6)
    assert moves2 == [] and all(np.allclose(out2[f][1], ground[f][1]) for f in range(21, 60))


def test_keypoint_confidence_keeps_the_best_camera_per_rotation():
    import numpy as np
    import pandas as pd
    from nfl_gsplat.render import play_timeline as pt, timeline as tlm
    rows = []
    for cam, wrist, elbow in (("sideline", 0.1, 0.9), ("endzone", 0.8, 0.2)):
        for joint in range(17):
            conf = {9: wrist, 7: elbow}.get(joint, 0.95)
            rows.append(dict(frame=100, cam=cam, global_player_id=4, joint=joint, x=0.0, y=0.0, conf=conf))
    kdf = pd.DataFrame(rows)
    both = pt.keypoint_confidence(kdf)[4][100]
    side = pt.keypoint_confidence(kdf, cam="sideline")[4][100]
    assert both.shape == (21,)
    assert abs(both[17] - 0.8) < 1e-9 and abs(side[17] - 0.1) < 1e-9      # the left elbow's rotation <- the left wrist (COCO 9)
    assert abs(both[15] - 0.9) < 1e-9 and abs(side[15] - 0.9) < 1e-9      # the left shoulder's <- the left elbow (COCO 7)
    assert np.isnan(both[2]) and np.isnan(both[11])                        # spine, neck: no keypoint vouches
    assert set(tlm.CONF_GATE_KEYPOINT) == {0, 1, 3, 4, 6, 7, 15, 16, 17, 18, 19, 20}
    assert pt.keypoint_confidence(kdf[kdf.cam == "nowhere"]) == {}
