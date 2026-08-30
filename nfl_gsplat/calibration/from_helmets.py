"""Cameras from labelled helmets and known player positions.

This is the calibration path the Helmet Assignment set opens up: the helmet
boxes are already assigned to a named player, and the tracking says where that
player was, so every frame supplies ~22 correspondences between the field and
the image with no detection, no tracking and no matching of our own in the way.

WHY A HOMOGRAPHY AND NOT A FULL SOLVE. One nominal helmet height makes the 3D
points COPLANAR, and a projection matrix is degenerate on coplanar points --
measured earlier, it returned an identical residual for offsets seven seconds
apart. Coplanar points determine a homography exactly, and a homography of a
known plane is enough: it yields the focal, and with the focal the pose.

WHY ONE FRAME IS NOT ENOUGH, and what to use instead. A homography per frame
fits beautifully -- 6.3 px -- and still yields a useless camera. Measured over
one play the per-frame focal ranged 2125 to 18430 px, a factor of 8.7.

The mechanism is CONDITIONING, not absence: on noiseless points a single frame
recovers the focal exactly, so it is wrong to say a telephoto view has no focal
in it. What happens is that at this geometry -- a long lens, far away, looking
at a shallow patch of players -- the focal is very weakly constrained, and the
~8 px that real helmet-height variation puts on every correspondence is
amplified into that 8.7x spread. Pooling does not rescue it; the spread is the
signal that the individual estimates are meaningless.

What works is ``cameras_fixed_centre``: a tripod camera pans but does not
translate, so ONE centre shared over the play ties the frames together and
makes the focal determined. Measured against tracking, triangulation error went
from 6.17 m per-frame to 0.13 m, and 0.15 m on players held out of the
calibration. ``cameras_for_view`` and ``pooled_focal`` are kept because they
are what establishes that per-frame fitting fails.

WHAT THE WORLD FRAME IS. z = 0 is the HELMET plane, not the ground, because
that is the plane the correspondences lie on. Its height above the turf never
enters: both cameras are recovered in the same frame, so triangulated x and y
compare directly against tracking, and z comes out as height relative to the
helmet plane -- which is a free check, since it should scatter by about the
0.3 m that real helmet heights vary.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.decompose_homography import (
    _solve_focal,
    homography_to_rt,
)
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Inlier distance for the plane fit, in pixels. Above the ~8 px floor that real
# helmet-height variation sets, below the cost of a genuinely wrong match.
RANSAC_PX: float = 10.0

# A homography needs four points; eight lets RANSAC reject a bad box instead of
# fitting it.
MIN_HELMETS: int = 8

# Frames needed before a pooled focal means anything.
MIN_FRAMES_FOR_FOCAL: int = 5

# Reprojection gate for the fixed-centre solve on HELMET correspondences.
# joint_solve's own default is 6 px, which is right for field-line
# intersections but impossible here: helmet height varies ~0.3 m, which is
# ~8 px, so even a perfect rigid camera cannot reproject helmets to 6 px. At
# the 6 px default the solve kept 0 of 48 frames; at 25 px, which is ~3x the
# correspondence noise, it keeps 48 of 48 and lands on a plausible camera.
HELMET_AUDIT_PX: float = 25.0


def plane_homography(world_xy, image_uv, *, ransac_px: float = RANSAC_PX):
    """``(H, inlier_mask)`` carrying field metres on the helmet plane to pixels."""
    import cv2

    world_xy = np.asarray(world_xy, dtype=np.float64)
    image_uv = np.asarray(image_uv, dtype=np.float64)
    if len(world_xy) < MIN_HELMETS:
        return None, np.zeros(len(world_xy), bool)
    H, mask = cv2.findHomography(world_xy, image_uv, cv2.RANSAC, ransac_px)
    if H is None:
        return None, np.zeros(len(world_xy), bool)
    return H, mask.ravel().astype(bool)


def pooled_focal(homographies, width: int, height: int, *,
                 min_frames: int = MIN_FRAMES_FOR_FOCAL):
    """``(focal, spread)`` over many frames of one camera.

    ``spread`` is the interquartile range, reported rather than hidden because
    a broadcast camera that zoomed mid-play has no single focal and the caller
    needs to know that happened instead of receiving an average of two.
    """
    cx, cy = width / 2.0, height / 2.0
    focals = []
    for H in homographies:
        if H is None:
            continue
        try:
            focals.append(_solve_focal(np.asarray(H, float), cx, cy))
        except ValueError:
            continue                      # degenerate pose: no focal in it
    if len(focals) < min_frames:
        raise CalibrationError(
            f"only {len(focals)} of {len(homographies)} frames yielded a focal, "
            f"need {min_frames}. The view is probably near-degenerate -- a "
            "camera whose plane axis is parallel to the image plane cannot "
            "reveal its focal. Try a play where the camera pans more.")
    focals = np.asarray(focals)
    return float(np.median(focals)), float(np.subtract(*np.percentile(focals, [75, 25])))


def cameras_for_view(byf, world_at, width: int, height: int, *,
                     ransac_px: float = RANSAC_PX,
                     min_frames: int = MIN_FRAMES_FOR_FOCAL):
    """``({frame: (K, R, t)}, focal)`` for one camera over a play.

    ``world_at(frame)`` returns the field positions, in metres, of every player
    the caller indexed -- so this function never needs to know about tracking
    clocks or offsets.

    Two passes on purpose: the focal is pooled over all frames first, then each
    frame's pose is solved with that focal held fixed. Letting each frame keep
    its own focal makes the recovered camera jitter for no physical reason.
    """
    per_frame = {}
    for frame in sorted(byf):
        uv, cols = byf[frame]
        world = np.asarray(world_at(frame), dtype=np.float64)[cols]
        ok = np.isfinite(world).all(axis=1)
        if ok.sum() < MIN_HELMETS:
            continue
        H, inliers = plane_homography(world[ok], uv[ok], ransac_px=ransac_px)
        if H is None:
            continue
        per_frame[frame] = (H, world[ok], uv[ok], inliers)

    focal, spread = pooled_focal([v[0] for v in per_frame.values()],
                                 width, height, min_frames=min_frames)
    K = np.array([[focal, 0, width / 2.0],
                  [0, focal, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    cams = {}
    for frame, (H, _w, _uv, _in) in per_frame.items():
        R, t = homography_to_rt(H, K)
        cams[frame] = (K, R, t)
    _LOG.info("recovered %d cameras, focal %.1f px (IQR %.1f)",
              len(cams), focal, spread)
    return cams, focal


def _arr(x):
    """CalibrationResult exposes some of these as methods and some as fields."""
    return np.asarray(x() if callable(x) else x, dtype=np.float64)


def cameras_fixed_centre(byf, world_at, width: int, height: int, *,
                         audit_px: float = HELMET_AUDIT_PX, view_deg: int = 0,
                         players=None, min_corrs: int = 6):
    """``({frame: (K, R, t)}, centre, mirrored)`` from ONE shared camera centre.

    Prefer this to ``cameras_for_view``. Fitting each frame independently does
    not work on this footage: at this geometry the focal is weakly conditioned,
    so the ~8 px that real helmet-height variation puts on every correspondence
    is amplified enormously. Measured on one play the per-frame focal ranged
    2125 to 18430 px -- a factor of 8.7 -- while the underlying homography
    fitted to 6.3 px. The fit was fine; the focal was not recoverable through
    the noise.

A tripod camera pans but does not translate, so one centre shared across
    the play ties the frames together and makes the focal determined. Measured
    against tracking on the same play, triangulation went from 6.17 m of error
    to 0.13 m, and to 0.15 m on players held out of the calibration entirely.

    ``players`` restricts the fit to a subset of player columns, which is how
    that held-out check is run: calibrate on half the squad and score the half
    the solve never saw.

    A CAVEAT ON THE CENTRE. Distance along the viewing axis trades off against
    focal length -- a camera further away with a longer lens looks almost the
    same -- so the returned centre is only good to something like 15%: two fits
    of the same play put the sideline camera at y = -102 m and y = -87 m.
    Triangulation is unharmed, because the projection is what is pinned down.
    Do not read the centre as a survey.
    """
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center

    frame_data = {}
    for frame, (uv, cols) in byf.items():
        world = np.asarray(world_at(frame), dtype=np.float64)[cols]
        keep = np.isfinite(world).all(axis=1)
        if players is not None:
            keep = keep & np.array([c in players for c in cols], dtype=bool)
        if keep.sum() < min_corrs:
            continue
        # z = 0 IS the helmet plane, so the height of that plane above the turf
        # never has to be known -- see the module docstring.
        frame_data[frame] = (np.c_[world[keep], np.zeros(int(keep.sum()))],
                             uv[keep])
    if not frame_data:
        raise CalibrationError(
            f"no frame had {min_corrs} usable helmets; nothing to calibrate.")

    results, mirrored = solve_fixed_center(
        {}, (width, height), init_results=[None] * (max(frame_data) + 1),
        _frame_data_override=frame_data, view_deg=view_deg,
        audit_drop_px=audit_px)

    cams = {}
    for frame in frame_data:
        r = results[frame]
        if r is None:
            continue
        cams[frame] = (_arr(r.intrinsics.K), _arr(r.pose.R),
                       _arr(r.pose.t).reshape(3))
    if not cams:
        raise CalibrationError("the joint solve kept no frames.")
    centre = np.median([_arr(results[f].pose.center_world)
                        for f in cams], axis=0)
    _LOG.info("fixed-centre solve kept %d/%d frames, centre %s, mirrored=%s",
              len(cams), len(frame_data), np.round(centre, 1), mirrored)
    return cams, centre, mirrored


def triangulate_matched(P_a, uv_a, cols_a, P_b, uv_b, cols_b):
    """``(player_columns, xyz)`` for the players BOTH views saw.

    Correspondence is by player identity, not by geometry. That is the whole
    reason this dataset is worth using: matching players across two views by
    appearance or epipolar geometry was measured three ways here and failed
    every time, with errors the size of the spacing between players.
    """
    import cv2

    cols_a = np.asarray(cols_a)
    cols_b = np.asarray(cols_b)
    shared = np.intersect1d(cols_a, cols_b)
    if len(shared) == 0:
        return shared, np.zeros((0, 3))
    ia = {c: i for i, c in enumerate(cols_a)}
    ib = {c: i for i, c in enumerate(cols_b)}
    pa = np.asarray(uv_a, dtype=np.float64)[[ia[c] for c in shared]]
    pb = np.asarray(uv_b, dtype=np.float64)[[ib[c] for c in shared]]
    h = cv2.triangulatePoints(np.asarray(P_a, dtype=np.float64),
                              np.asarray(P_b, dtype=np.float64),
                              pa.T.copy(), pb.T.copy())
    return shared, (h[:3] / h[3]).T
