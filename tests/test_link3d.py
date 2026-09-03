"""Linking per-frame ground placements into players, on the field plane.

Measured on the helmet set (07j), BoT-SORT per view breaks each player into
5-7 pieces per play with a tenth-percentile purity under 0.6. These tests are
the bar the ground-plane linker has to clear before it replaces that.
"""
import numpy as np

from nfl_gsplat.tracking.link3d import assignments, link, smooth

FPS = 59.94


def synthetic_play(n_players=22, n_frames=90, seed=0, noise_m=0.3, dropout=0.1,
                   gap=None):
    """Players on the turf with steady velocities; two pairs brush past.

    Returns ``(placements, labels, truth)``: placements maps frame -> [N, 2],
    labels maps frame -> [N] team ids, truth maps frame -> [P, 2] (NaN where a
    player is dropped). Teams alternate by index. The crossing pairs are on
    opposite teams and pass 0.8 m apart -- players do not run through each
    other, and two identical points that do are a coin flip for anything.
    """
    rng = np.random.default_rng(seed)
    # Football-like, not a random gas: spread over 40 x 30 m, and a third of
    # the players (linemen) barely move. Twenty-two random walkers in a
    # 30 x 24 m box manufacture same-team passes within half a metre at
    # speed every second -- the one thing position-only linking cannot
    # settle -- far more often than a play does.
    start = rng.uniform([-20, -15], [20, 15], size=(n_players, 2))
    vel = rng.uniform(-7.0, 7.0, size=(n_players, 2))
    vel[rng.random(n_players) < 0.33] *= 0.05
    team = np.arange(n_players) % 2
    if n_players >= 2:
        start[0], start[1] = [-5.0, 0.0], [5.0, 0.8]
        vel[0], vel[1] = [7.0, 0.0], [-7.0, 0.0]
    if n_players >= 4:
        # Same sideways drift, so the 0.8 m gap holds through the crossing.
        start[2], start[3] = [0.0, -5.0], [0.8, 5.0]
        vel[2], vel[3] = [0.5, 7.0], [0.5, -7.0]
    placements, labels, truth = {}, {}, {}
    for f in range(n_frames):
        xy = start + vel * (f / FPS)
        keep = rng.random(n_players) >= dropout
        if gap is not None:
            g0, g1, who = gap
            if g0 <= f < g1:
                keep[who] = False
        seen = xy[keep] + rng.normal(0.0, noise_m, (int(keep.sum()), 2))
        order = rng.permutation(len(seen))
        placements[f] = seen[order]
        labels[f] = team[keep][order]
        t = xy.copy()
        t[~keep] = np.nan
        truth[f] = t
    return placements, labels, truth


def purity_and_fragments(tracks, truth, gate_m=1.0):
    """Per track: share of its points nearest its MAJORITY player; and, per
    player, how many tracks have that player as their majority (fragments).

    Majority, not "any point": two players brushing past put a few points of
    a perfect track nearer the other player, and counting those as pieces
    would fail a tracker for the synthetic's own geometry.
    """
    purities, pieces = [], {}
    for tr in tracks:
        assigned = []
        for f, xy in zip(tr.frames, tr.xy):
            t = truth[f]
            ok = np.isfinite(t).all(1)
            d = np.linalg.norm(t[ok] - xy, axis=1)
            if len(d) and d.min() <= gate_m:
                assigned.append(int(np.flatnonzero(ok)[int(np.argmin(d))]))
        if not assigned:
            continue
        counts = np.bincount(assigned)
        purities.append(counts.max() / len(assigned))
        pieces.setdefault(int(counts.argmax()), set()).add(tr.id)
    frags = [len(v) for v in pieces.values()]
    return np.asarray(purities), np.asarray(frags)


def test_every_player_becomes_one_pure_track():
    placements, labels, truth = synthetic_play()
    tracks = link(placements, labels=labels, fps=FPS)
    long = [t for t in tracks if len(t.frames) >= 45]
    assert len(long) == 22
    purity, frags = purity_and_fragments(tracks, truth)
    assert np.median(purity) == 1.0
    # Same-team close passes at speed remain a genuine ambiguity for
    # position-only linking; the real-data number is what 07j measures.
    assert np.percentile(purity, 10) >= 0.9
    assert np.median(frags) == 1


def test_crossing_players_keep_their_identities():
    """Two players brushing past each other must not swap."""
    placements, labels, truth = synthetic_play(n_players=4, noise_m=0.15, dropout=0.0)
    tracks = link(placements, labels=labels, fps=FPS)
    purity, frags = purity_and_fragments(tracks, truth)
    assert len(tracks) == 4
    # At a 0.8 m pass with 0.15 m noise a few points of a correct track sit
    # nearer the other player; the majority owner is what identity is.
    assert purity.min() >= 0.95
    assert frags.max() == 1


def test_a_short_gap_is_bridged_and_a_long_one_is_not():
    short = synthetic_play(n_players=6, dropout=0.0, gap=(30, 48, 5))    # 0.3 s
    tracks = link(short[0], labels=short[1], fps=FPS)
    _p, frags = purity_and_fragments(tracks, short[2])
    assert frags.max() == 1

    long_gap = synthetic_play(n_players=6, dropout=0.0, gap=(20, 80, 5))  # 1.0 s
    tracks = link(long_gap[0], labels=long_gap[1], fps=FPS)
    _p, frags = purity_and_fragments(tracks, long_gap[2])
    assert frags.max() == 2                          # honest: two pieces


def test_smoothing_reduces_noise_without_lagging():
    placements, labels, truth = synthetic_play(n_players=3, noise_m=0.35, dropout=0.0)
    tracks = link(placements, labels=labels, fps=FPS)
    for tr in tracks:
        raw = np.asarray(tr.xy)
        sm = smooth(tr, fps=FPS)
        t = np.array([truth[f][np.nanargmin(np.linalg.norm(truth[f] - raw[i], axis=1))]
                      for i, f in enumerate(tr.frames)])
        err_raw = np.sqrt(np.mean(np.sum((raw - t) ** 2, axis=1)))
        err_sm = np.sqrt(np.mean(np.sum((sm - t) ** 2, axis=1)))
        assert err_sm < 0.6 * err_raw


def test_without_labels_players_that_stay_apart_still_link():
    """Labels are an aid, not a requirement."""
    placements, _labels, truth = synthetic_play(n_players=8, seed=3, dropout=0.0)
    # Drop the two brushing pairs; keep the six that wander apart.
    for f in placements:
        far = np.linalg.norm(placements[f][:, None] - truth[f][None, :4], axis=2).min(1) > 2.0
        placements[f] = placements[f][far]
    tracks = link(placements, fps=FPS)
    purity, _frags = purity_and_fragments(tracks, truth)
    assert np.median(purity) == 1.0


def test_clutter_does_not_become_a_player():
    placements, labels, _truth = synthetic_play(n_players=5, dropout=0.0)
    rng = np.random.default_rng(1)
    for f in placements:                             # one stray point per frame
        placements[f] = np.vstack([placements[f], rng.uniform(-40, 40, 2)])
        labels[f] = np.append(labels[f], -1)
    tracks = link(placements, labels=labels, fps=FPS)
    assert len([t for t in tracks if len(t.frames) >= 45]) == 5


def test_assignments_are_index_aligned_even_for_duplicate_points():
    """Two detections on the same ground point in one frame (an NMS miss)
    must each keep their own row; a lookup by value gave both the id written
    last (review finding)."""
    placements, labels, _truth = synthetic_play(n_players=4, dropout=0.0)
    f0 = min(placements)
    dup = np.vstack([placements[f0], placements[f0][:1]])   # duplicate row 0
    placements[f0] = dup
    labels[f0] = np.append(labels[f0], labels[f0][0])
    tracks = link(placements, labels=labels, fps=FPS)
    ids = assignments(tracks, placements)
    assert len(ids[f0]) == len(dup)
    claimed = ids[f0][ids[f0] >= 0]
    assert len(claimed) == len(set(claimed))                # one row per track
    for tr in tracks:                                        # rows round-trip
        for f, row, xy in zip(tr.frames, tr.rows, tr.xy):
            assert np.allclose(placements[f][row], xy)


def test_appearance_breaks_the_ties_labels_could_not():
    """Two brushing pairs, same-team-looking, NO labels: position alone may
    swap them; a per-player appearance vector (with noise) must keep them."""
    placements, _labels, truth = synthetic_play(n_players=4, noise_m=0.3, dropout=0.0)
    rng = np.random.default_rng(7)
    ident = rng.normal(size=(4, 32))
    ident /= np.linalg.norm(ident, axis=1, keepdims=True)
    features = {}
    for f, pts in placements.items():
        # Which true player each placement is (nearest truth), then its vector
        # plus noise -- crops of one player vary, but less than between players.
        who = np.array([int(np.nanargmin(np.linalg.norm(truth[f] - p, axis=1))) for p in pts])
        feat = ident[who] + rng.normal(scale=0.35, size=(len(pts), 32))
        features[f] = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    tracks = link(placements, features=features, fps=FPS, feature_weight=1.0)
    purity, frags = purity_and_fragments(tracks, truth)
    assert len(tracks) == 4
    assert purity.min() >= 0.95
    assert frags.max() == 1

