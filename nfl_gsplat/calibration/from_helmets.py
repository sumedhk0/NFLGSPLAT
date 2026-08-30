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

WHY THE FOCAL IS POOLED. A single frame's focal comes only from that frame's
plane orientation, so it is noisy, and at particular poses it is not determined
at all -- a camera whose plane axis is parallel to the image plane, which is
roughly the nominal endzone pose. Broadcast cameras do zoom, so the focal is
not constant forever; ``focal_spread`` reports how much it moved so a play
where it did can be caught rather than averaged into nonsense.

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
