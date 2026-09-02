"""The players must be able to calibrate the view the paint cannot -- a tight,
panning, zooming endzone camera that never holds the whole formation."""
import numpy as np
import pytest

from nfl_gsplat.calibration.from_players import (
    fit_frame,
    focal_from_boxes,
    look_at,
    solve_second_view,
)
from nfl_gsplat.errors import CalibrationError

W, H = 1920, 1080
PLAYER_M = 1.85


def cam(centre, f, target=(6.0, 4.0, 0.0)):
    R = look_at(centre, target)
    t = -R @ np.asarray(centre, float)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, R, t


def project(c, pts):
    K, R, t = c
    q = (np.asarray(pts, float) @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:3]


def boxes_for(c, xy, height_m=PLAYER_M, half_w_px=15.0):
    """Person boxes as the detector draws them, for the players IN the frame."""
    feet = project(c, np.c_[xy, np.zeros(len(xy))])
    head = project(c, np.c_[xy, np.full(len(xy), height_m)])
    b = np.column_stack([feet[:, 0] - half_w_px, head[:, 1],
                         feet[:, 0] + half_w_px, feet[:, 1]])
    inside = ((b[:, 0] >= 0) & (b[:, 2] < W) & (b[:, 1] >= 0) & (b[:, 3] < H))
    return b[inside]


# The sideline camera the players verified on the real play (12 deg lens), and
# an endzone mount like the real one: behind the +x end zone, low-ish, steep
# enough to look down on the box, behind a long lens.
SIDE = cam([29.5, -95.3, 47.1], 7500.0)
ENDZ_CENTRE = np.array([80.0, -3.0, 22.0])


def play(n_frames=8, n_players=20, seed=0, head_m=PLAYER_M, side_stretch=1.0):
    """A formation seen by the sideline and by a panning, zooming endzone."""
    rng = np.random.default_rng(seed)
    start = rng.uniform([-4, -3], [16, 11], size=(n_players, 2))
    feet_a, boxes_b, truth, cams_b = {}, {}, {}, {}
    for i in range(n_frames):
        xy = start + rng.normal(0, 0.5, start.shape) + i * np.array([0.5, 0.2])
        f = 100 + 30 * i
        truth[f] = xy
        # What the sideline reports, possibly of a mislabelled (stretched) world.
        seen = xy * np.array([1.0, side_stretch])
        feet_a[f] = project(SIDE, np.c_[seen, np.zeros(len(seen))])
        # The endzone zooms and pans: lens 9000..11000 px, aim drifts with play.
        f_e = 9000.0 + 250.0 * i
        aim = np.r_[xy.mean(0) + [8.0 - 0.6 * i, 4.0 + 0.3 * i], 0.0]
        cams_b[f] = cam(ENDZ_CENTRE, f_e, target=aim)
        boxes_b[f] = boxes_for(cams_b[f], xy, height_m=head_m)
    return feet_a, boxes_b, truth, cams_b


FAST_MOUNTS = [(sx * x, 0.0, z) for sx in (-1, 1)
               for x, z in ((60, 20), (80, 35), (95, 50))]


def test_the_tight_endzone_never_holds_the_whole_formation():
    """Sanity on the synthetic: it must be as hard as the real thing."""
    _fa, boxes_b, truth, _cb = play()
    seen = np.median([len(boxes_b[f]) for f in boxes_b])
    assert 8 <= seen < len(truth[100])


def test_the_boxes_read_the_lens_even_at_a_steep_pitch():
    """A 1.85 m player at range d, seen from pitch p, is ~f*1.85*cos(p)/d px."""
    _fa, boxes_b, truth, cams_b = play()
    f0 = 100
    xy = truth[f0].mean(0)
    d = np.linalg.norm(ENDZ_CENTRE - np.r_[xy, 0.0])
    pitch = np.degrees(np.arctan2(ENDZ_CENTRE[2], np.linalg.norm(ENDZ_CENTRE[:2] - xy)))
    f_true = cams_b[f0][0][0, 0]
    f_hat = focal_from_boxes({f0: boxes_b[f0]}, d, pitch_deg=pitch)
    assert abs(f_hat - f_true) / f_true < 0.15


def test_it_finds_the_mount_from_the_full_grid():
    feet_a, boxes_b, _truth, _cb = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    cams, info = solve_second_view(cams_a, feet_a, boxes_b, W, H)
    assert info["centre"][0] > 30.0            # behind the +x end zone
    # The refinement sweeps y; the true mount is at y = -3.
    assert abs(info["centre"][1] - (-3.0)) <= 5.0
    assert info["reconciled"] >= 8
    # The centre is a prior from a coarse grid; the nearest grid mount is 5 m
    # and 3 m off the truth and that costs about 0.45 m in the plane map on
    # noiseless data. Measured, and the price of an unobservable distance.
    assert info["gap_m"] < 0.6
    assert len(cams) >= 6


def test_per_frame_cameras_follow_the_pan_and_zoom():
    """The point of per-frame cameras: each frame must agree with ITS truth."""
    feet_a, boxes_b, truth, cams_b = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    # Seeds near the true mount: this test is about the per-frame pan and
    # zoom, not the centre prior. (The prior's cost is measured separately:
    # a seed 13 m off in height put the plane map ~40 px, 0.3 m, off.)
    cams, _info = solve_second_view(cams_a, feet_a, boxes_b, W, H,
                                    mounts=[(80.0, 0.0, 20.0), (-80.0, 0.0, 20.0)],
                                    refine=False)
    errs = []
    for f, cam_hat in cams.items():
        pts = np.c_[truth[f], np.zeros(len(truth[f]))]
        want = project(cams_b[f], pts)
        got = project(cam_hat, pts)
        inside = (want[:, 0] >= 0) & (want[:, 0] < W) & (want[:, 1] >= 0) & (want[:, 1] < H)
        errs.append(np.median(np.linalg.norm(want[inside] - got[inside], axis=1)))
    assert np.median(errs) < 20.0             # px, at f ~ 10000 that is ~0.15 m


def test_a_camera_that_fits_the_feet_but_not_the_people_is_refused():
    """Feet consistent, heads say 5 m: perfect fit, wrong world, refused."""
    # 3.2 m, not 5: a 5 m head projects above the top of the frame and the
    # detector would not draw the box at all.
    feet_a, boxes_b, _truth, _cb = play(head_m=3.2)
    cams_a = dict.fromkeys(feet_a, SIDE)
    with pytest.raises(CalibrationError, match="player cost"):
        solve_second_view(cams_a, feet_a, boxes_b, W, H, mounts=FAST_MOUNTS,
                          refine=False)


def test_two_views_of_different_worlds_are_refused():
    """The measured failure: a sideline whose Y is stretched threefold.

    The endzone sees the real formation; the sideline reports one three times
    as wide. No mount reconciles them, and it must say so rather than return
    the least bad camera.
    """
    feet_a, boxes_b, _truth, _cb = play(side_stretch=3.0)
    cams_a = dict.fromkeys(feet_a, SIDE)
    with pytest.raises(CalibrationError, match="could not calibrate"):
        solve_second_view(cams_a, feet_a, boxes_b, W, H, mounts=FAST_MOUNTS,
                          refine=False)


def test_the_aim_search_finds_a_frame_centred_off_the_formation():
    """A seed pointed at the formation centroid sees a different patch of turf
    than the tight endzone frame does; without the search nothing matches."""
    _fa, boxes_b, truth, cams_b = play()
    f = 100
    K_true = cams_b[f][0]
    feet_b = np.column_stack([(boxes_b[f][:, 0] + boxes_b[f][:, 2]) / 2.0,
                              boxes_b[f][:, 3]])
    formation = truth[f].mean(0)
    d = np.linalg.norm(ENDZ_CENTRE - np.r_[formation, 0.0])
    out = fit_frame(ENDZ_CENTRE, K_true, truth[f], feet_b, formation,
                    K_true[0, 0] / d)
    assert out is not None
    _R, _t, n = out
    assert n >= 0.8 * len(feet_b)


def test_the_fitted_lens_makes_the_players_the_right_height():
    """The fitted lens must make the players the right height.

    Guards a property, not a fix. The endzone aims at the box, about eight
    metres nearer than the formation centroid the ruler's range is taken
    from; on this synthetic that costs under 5%, and it passed before any
    change was made. The real play's endzone heights come out 1.45 m, a 22%
    deficit this synthetic does NOT reproduce -- the leading suspect is the
    sideline's own range ambiguity along its axis (its centre is only good to
    ~15%), which would move the whole formation toward or away from the
    endzone. Open; recorded here so nobody mistakes this test for its fix.
    """
    from nfl_gsplat.calibration.player_scale import implied_heights

    feet_a, boxes_b, _truth, _cb = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    cams, _info = solve_second_view(cams_a, feet_a, boxes_b, W, H,
                                    mounts=[(80.0, 0.0, 20.0)], refine=False)
    hs = np.concatenate([implied_heights(*cams[f], boxes_b[f]) for f in cams])
    hs = hs[(hs > 0.5) & (hs < 4.0)]
    assert abs(np.median(hs) - PLAYER_M) / PLAYER_M < 0.05
