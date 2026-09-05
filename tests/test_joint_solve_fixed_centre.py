"""joint_solve.solve_fixed_center(fixed_center=...): the mount held, the lens from the paint."""
import numpy as np

from nfl_gsplat.calibration import joint_solve as js


def _camera_frames(C, focals, rng):
    """Synthetic tripod: one centre, per-frame focal and a small pan; ground
    landmarks on a 5 m grid projected into a 1920x1080 frame."""
    image_size = (1920, 1080)
    gx, gy = np.meshgrid(np.arange(-30.0, 31.0, 5.0), np.arange(-25.0, 26.0, 5.0))
    world = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    frame_data = {}
    truth = {}
    for i, f in enumerate(focals):
        target = np.array([2.0 * i - 4.0, 0.0, 0.0])
        R = js._look_at_R(C, target)
        r = _rodrigues(R)
        uv = js._project(world, r, f, C, image_size)
        keep = (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
        frame_data[i] = (world[keep], uv[keep] + rng.normal(0, 0.5, size=(keep.sum(), 2)))
        truth[i] = f
    return frame_data, image_size, truth


def _rodrigues(R):
    import cv2

    return cv2.Rodrigues(np.asarray(R, float))[0].reshape(3)


def test_fixed_centre_recovers_the_lens_per_frame():
    rng = np.random.default_rng(0)
    C = np.array([-4.0, -100.0, 42.0])
    focals = [9000.0 + 300.0 * i for i in range(12)]
    frame_data, image_size, truth = _camera_frames(C, focals, rng)
    results, mirrored = js.solve_fixed_center(
        {}, image_size, init_results=[None] * len(focals), _frame_data_override=frame_data,
        view_deg=0, audit_drop_px=3.0, fixed_center=C)
    got = [r for r in results if r is not None]
    assert len(got) >= 10
    for i, r in enumerate(results):
        if r is None:
            continue
        f = float(r.intrinsics.fx)
        assert abs(f / truth[i] - 1.0) < 0.02, (i, f, truth[i])
