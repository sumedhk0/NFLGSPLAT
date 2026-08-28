"""End-to-end identity wiring: grounding, stitching, votes, formation, merge."""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.identity.resolve import (ground_tracks, resolve_camera,
                                         resolve_play)


def _track(n_frames=40, conf=None):
    """A camera 80 m back and 30 m up, looking down the field."""
    k = np.array([[1200.0, 0, 960.0], [0, 1200.0, 540.0], [0, 0, 1.0]])
    rot = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    centre = np.array([-80.0, 0.0, 30.0])
    t = -rot @ centre
    return CameraTrack(K=np.tile(k, (n_frames, 1, 1)),
                       R=np.tile(rot, (n_frames, 1, 1)),
                       t=np.tile(t, (n_frames, 1)),
                       conf=(np.ones(n_frames) if conf is None
                             else np.asarray(conf, float)),
                       width=1920, height=1080)


def _project(track, frame, xyz):
    intr, pose = track.at(frame)
    cam = pose.R @ np.asarray(xyz, float) + pose.t
    uv = intr.K() @ cam
    return uv[0] / uv[2], uv[1] / uv[2]


def _tracks_df(track, rows):
    """rows: list of (frame, track_id, x_m, y_m)."""
    out = []
    for frame, tid, x, y in rows:
        u, v = _project(track, frame, [x, y, 0.0])
        out.append({"cam": "sideline", "frame": frame, "track_id": tid,
                    "foot_u": u, "foot_v": v,
                    "bbox_x1": u - 20, "bbox_y1": v - 80,
                    "bbox_x2": u + 20, "bbox_y2": v})
    return pd.DataFrame(out)


def _on_field():
    return pd.DataFrame({
        "team": ["ARI", "ARI", "SEA"],
        "jersey_number": [14.0, 4.0, 21.0],
        "full_name": ["Michael Wilson", "Greg Dortch", "Devon Witherspoon"],
        "position": ["WR", "WR", "DB"],
        "height_m": [1.91, 1.70, 1.83],
        "weight": [200.0, 173.0, 180.0],
    })


def test_grounding_recovers_the_world_position_it_was_projected_from():
    track = _track()
    df = _tracks_df(track, [(5, 1, -20.0, 3.0), (5, 2, -10.0, -6.0)])
    positions, by_frame = ground_tracks(track, df, "sideline")
    got = {t: rows[0][1:] for t, rows in positions.items()}
    assert got[1] == pytest.approx((-20.0, 3.0), abs=1e-6)
    assert got[2] == pytest.approx((-10.0, -6.0), abs=1e-6)
    ids, xy = by_frame[5]
    assert sorted(ids) == [1, 2] and xy.shape == (2, 3 - 1)


def test_unverified_frames_are_never_grounded():
    """conf == 0 means the pose was interpolated, so its positions are guesses."""
    conf = np.ones(40)
    conf[5] = 0.0
    track = _track(conf=conf)
    df = _tracks_df(track, [(5, 1, -20.0, 3.0), (6, 1, -20.0, 3.0)])
    positions, by_frame = ground_tracks(track, df, "sideline")
    assert 5 not in by_frame
    assert [r[0] for r in positions[1]] == [6]


def test_fragments_are_stitched_and_their_votes_pooled():
    """The reason stitching exists: numbers read late, alignment reads early,
    and the tracker splits them onto different ids."""
    track = _track()
    rows = ([(f, 1, -20.0 + 0.05 * f, 2.0) for f in range(0, 10)]
            + [(f, 2, -20.0 + 0.05 * f, 2.0) for f in range(11, 30)])
    df = _tracks_df(track, rows)
    votes = {1: collections.Counter({14: 3}), 2: collections.Counter({14: 6})}
    idents, positions, player_of = resolve_camera(
        track, df, "sideline", _on_field(), votes)
    assert player_of[2] == player_of[1]          # one player, not two
    assert len(idents) == 1
    assert idents[0].jersey == 14
    assert idents[0].votes == 9                  # 3 + 6, pooled across fragments
    assert len(positions[player_of[1]]) == 29    # both fragments' frames


def test_truncated_reads_are_credited_through_the_pipeline():
    """53 for #14 against 36 for #4 fails the margin rule until 4 is credited."""
    track = _track()
    df = _tracks_df(track, [(f, 1, -20.0, 2.0) for f in range(20)])
    votes = {1: collections.Counter({14: 53, 4: 36})}
    idents, _pos, _p = resolve_camera(track, df, "sideline", _on_field(), votes,
                                      truncation_credit=0.0)
    assert idents == []
    idents, _pos, _p = resolve_camera(track, df, "sideline", _on_field(), votes,
                                      truncation_credit=1.0)
    assert [i.jersey for i in idents] == [14]


def test_resolve_play_drops_a_pairing_geometry_refutes():
    from nfl_gsplat.identity.jersey_vote import TrackIdentity
    a = TrackIdentity(track_id=1, jersey=14, player="Michael Wilson",
                      team="ARI", height_m=1.91, votes=4, margin=3.0,
                      weight_lb=200.0)
    b = TrackIdentity(track_id=7, jersey=14, player="Michael Wilson",
                      team="ARI", height_m=1.91, votes=40, margin=3.0,
                      weight_lb=200.0)
    far = {f: np.array([30.0, 0.0]) for f in range(40)}
    near = {f: np.array([0.0, 0.0]) for f in range(40)}
    merged, checks = resolve_play({"sideline": ([a], {1: near}),
                                   "endzone": ([b], {7: far})})
    assert checks[14].contradicted
    assert merged[14].cameras == ("endzone",)     # the better-evidenced camera
