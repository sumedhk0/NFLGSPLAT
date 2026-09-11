"""pair_tracks: per-camera tracks paired by their mean offset over the overlap; fragments, clip lag, kit veto."""
import numpy as np

from nfl_gsplat.tracking import link3d
from nfl_gsplat.tracking.pair_tracks import global_ids, pair_tracks

FPS = 60.0


def _walk(rng, n_players=6, n_frames=120):
    """Players 1.2 m apart walking along x with a little wander; two cameras
    see the same walk with ANISOTROPIC noise: the sideline blurs y (its
    depth) by 1.0 m, the endzone blurs x by 1.0 m, both 0.15 m the other way.
    The endzone clip lags by 3 frames."""
    truth = {}
    for p in range(n_players):
        x0, y0 = rng.uniform(-10, 10), 1.2 * p
        v = rng.uniform(0.02, 0.06)
        truth[p] = np.array([[x0 + v * f + 0.3 * np.sin(f / 20 + p), y0 + 0.2 * np.cos(f / 25 + p)]
                             for f in range(n_frames)])
    return truth


def _tracks(truth, rng, *, sx, sy, lag=0, split=None, label=None):
    """Track3D per player (optionally split into fragments at ``split``
    frames); each point = truth + noise; ``lag`` shifts the frame numbers."""
    out = []
    tid = 0
    for p, xy in truth.items():
        n = len(xy)
        cuts = [0] + (split.get(p, []) if split else []) + [n]
        for a, b in zip(cuts[:-1], cuts[1:]):
            t = link3d.Track3D(id=tid)
            tid += 1
            for f in range(a, b):
                noisy = xy[f] + rng.normal(0, [sx, sy])
                t.frames.append(f + lag)
                t.xy.append(noisy)
                t.labels.append(label[p] if label else -1)
                t.rows.append(0)
            out.append(t)
    return out


def test_pairs_recover_players_fragments_and_the_clip_lag():
    rng = np.random.default_rng(0)
    truth = _walk(rng)
    side = _tracks(truth, rng, sx=0.15, sy=1.0, split={1: [50], 3: [40, 80]})
    end = _tracks(truth, rng, sx=1.0, sy=0.15, lag=3, split={2: [60]})
    pairs, k = pair_tracks(side, end, fps=FPS)
    assert abs(k - 3) <= 1, k        # a frame either way is inside the noise at 60 Hz
    gid_s, gid_e = global_ids(len(side), len(end), pairs)
    # every sideline fragment of a player shares its id with that player's endzone track(s)
    owner_s = [p for p, xy in truth.items() for _ in ([0] + ({1: [50], 3: [40, 80]}.get(p, [])))]
    owner_e = [p for p, xy in truth.items() for _ in ([0] + ({2: [60]}.get(p, [])))]
    for i, p in enumerate(owner_s):
        js = [j for j, q in enumerate(owner_e) if q == p]
        assert all(gid_e[j] == gid_s[i] for j in js), (i, p)
    # and no two players share an id
    ids_by_player = {}
    for i, p in enumerate(owner_s):
        ids_by_player.setdefault(p, set()).add(int(gid_s[i]))
    all_ids = [g for s in ids_by_player.values() for g in s]
    assert len(all_ids) == len(set(all_ids))


def test_kit_veto_blocks_a_cross_kit_pair_even_when_it_is_the_nearest():
    rng = np.random.default_rng(1)
    truth = _walk(rng, n_players=2)
    truth[1] = truth[0] + np.array([0.3, 0.0])            # two players 0.3 m apart the whole time
    side = _tracks(truth, rng, sx=0.15, sy=1.0, label={0: 1, 1: 0})
    end = _tracks(truth, rng, sx=1.0, sy=0.15, label={0: 1, 1: 0})
    pairs, _ = pair_tracks(side, end, fps=FPS, frame_offsets=[0])
    assert {(p.i, p.j) for p in pairs} == {(0, 0), (1, 1)}
    pairs_no_veto, _ = pair_tracks(side, end, fps=FPS, frame_offsets=[0], kit_veto=False)
    assert len(pairs_no_veto) == 2                         # still one-to-one, kit or not
