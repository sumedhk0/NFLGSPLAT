"""View-rotation utilities. The pixel map is validated against cv2.rotate
itself (single marked pixel), not against a hand-derived formula."""
import numpy as np
import pytest

from nfl_gsplat.calibration.view_rotation import (
    derotate_result, rotate_image, rotate_uv, rotated_wh,
)
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

W, H = 32, 20                       # small asymmetric test image


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_rotate_uv_matches_cv2_rotate(deg):
    # mark one pixel, rotate the image with cv2, find where it went, and
    # demand rotate_uv predicts exactly that location
    img = np.zeros((H, W), np.uint8)
    u, v = 5, 3
    img[v, u] = 255
    rot = rotate_image(img, deg)
    ys, xs = np.nonzero(rot)
    assert len(xs) == 1
    pu, pv = rotate_uv(float(u), float(v), deg, (W, H))
    assert (round(pu), round(pv)) == (xs[0], ys[0])
    assert rot.shape[::-1] == rotated_wh(deg, (W, H))


def test_rotate_uv_round_trip_90_270():
    u2, v2 = rotate_uv(5.0, 3.0, 90, (W, H))
    u3, v3 = rotate_uv(u2, v2, 270, rotated_wh(90, (W, H)))
    assert (u3, v3) == (5.0, 3.0)


def test_rotate_image_invalid_deg_raises():
    with pytest.raises(ValueError, match="rotation"):
        rotate_image(np.zeros((4, 4), np.uint8), 45)


def _camera_for_rotated(deg, f=1400.0):
    """A camera solved IN the rotated frame of a (1920,1080) original, looking
    at the field plane (reuses the joint-solve test geometry)."""
    ow, oh = 1920, 1080
    rw, rh = rotated_wh(deg, (ow, oh))
    C = np.array([-90.0, 0.0, 35.0])
    target = np.array([-30.0, 0.0, 0.0])
    fwd = target - C; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    return CalibrationResult(
        intrinsics=CameraIntrinsics(f, f, rw / 2, rh / 2, rw, rh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.5,
        num_correspondences=8, refined_with_ba=True), (ow, oh)


@pytest.mark.parametrize("deg", [90, 180, 270])
def test_derotate_result_projection_equivalence(deg):
    # For any world point: projecting through the DE-ROTATED camera onto the
    # original image must equal mapping the rotated-camera projection back
    # through the pixel map. Tolerance 1.0 px: cv2.rotate maps pixel INDICES
    # (0..N-1) while intrinsics use the W/2 center convention — a constant
    # half-pixel offset, not an accumulating error.
    res_rot, orig_wh = _camera_for_rotated(deg)
    res = derotate_result(res_rot, deg, orig_wh)
    pts = np.array([[-30.0, 5.0, 0.0], [-40.0, -10.0, 0.0], [-20.0, 0.0, 0.0]])
    uv_rot = project_points(pts, res_rot.intrinsics.K(), res_rot.pose.R, res_rot.pose.t)
    uv_orig = project_points(pts, res.intrinsics.K(), res.pose.R, res.pose.t)
    inv = {90: 270, 180: 180, 270: 90}[deg]
    rw_h = rotated_wh(deg, orig_wh)
    mapped = np.array([rotate_uv(u, v, inv, rw_h) for (u, v) in uv_rot])
    assert np.abs(mapped - uv_orig).max() <= 1.0
    # camera center is rotation-invariant
    assert np.allclose(res.pose.center_world(), res_rot.pose.center_world(), atol=1e-9)
    # intrinsics rebuilt for the original dims
    assert (res.intrinsics.width, res.intrinsics.height) == orig_wh
    assert res.intrinsics.fx == res_rot.intrinsics.fx


def test_derotate_zero_is_identity():
    res_rot, orig_wh = _camera_for_rotated(0)
    res = derotate_result(res_rot, 0, orig_wh)
    assert np.allclose(res.pose.R, res_rot.pose.R)
    assert np.allclose(res.pose.t, res_rot.pose.t)
