"""calibration.endzone_paint on synthetic geometry: registration, dash cleaning, per-frame fit, the centre."""
import numpy as np
import pytest

from nfl_gsplat.calibration import endzone_paint as ep
from nfl_gsplat.calibration.field_landmarks import GOAL_LINE_X_M, HASH_OFFSET_M, YARD_LINE_SPACING_M


def look_at(centre, target):
    """World -> camera rotation (OpenCV: z forward, y down) for a camera at ``centre`` aimed at ``target``."""
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    fwd = target - centre
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def camera(centre=(88.0, 0.6, 21.0), target=(-20.0, 0.0, 0.0), f=15000.0):
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    R = look_at(centre, target)
    t = -R @ np.asarray(centre, float)
    return K, R, t


def project(K, R, t, X):
    p = (K @ (R @ np.asarray(X, float).T + t[:, None])).T
    return p[:, :2] / p[:, 2:3]


def synthetic_frame(K, R, t, *, ks=range(0, 9), noise=0.0, seed=0):
    """A PaintFrame as the true camera would see it: yard-line segments k in ``ks`` (two
    segments each, either side of the middle), hash dashes every yard along both rows."""
    rng = np.random.default_rng(seed)
    p0s, p1s, kk = [], [], []
    for k in ks:
        x = -GOAL_LINE_X_M + YARD_LINE_SPACING_M * k
        for ya, yb in ((-7.0, -1.0), (1.0, 7.0)):
            a, b = project(K, R, t, [[x, ya, 0.0], [x, yb, 0.0]])
            p0s.append([a[0], a[1], 1.0])
            p1s.append([b[0], b[1], 1.0])
            kk.append(k)
    dashes, rows = [], []
    for r, y in enumerate((-HASH_OFFSET_M, HASH_OFFSET_M)):
        for x in np.arange(-44.0, -6.0, 0.9144):
            u = project(K, R, t, [[x, y, 0.0]])[0]
            dashes.append([u[0], u[1], 1.0])
            rows.append(r)
    P0, P1, D = np.asarray(p0s), np.asarray(p1s), np.asarray(dashes)
    if noise:
        P0[:, :2] += rng.normal(0, noise, P0[:, :2].shape)
        P1[:, :2] += rng.normal(0, noise, P1[:, :2].shape)
        D[:, :2] += rng.normal(0, noise, D[:, :2].shape)
    return ep.PaintFrame(P0, P1, np.asarray(kk, int), D, np.asarray(rows, int), None, len(list(ks)))


def perturb(K, R, centre, *, deg=1.5, zoom=0.08, axis=(0.3, 1.0, 0.2)):
    from scipy.spatial.transform import Rotation

    ax = np.asarray(axis, float)
    ax /= np.linalg.norm(ax)
    Rp = Rotation.from_rotvec(np.radians(deg) * ax).as_matrix() @ R
    Kp = K.copy()
    Kp[0, 0] = Kp[1, 1] = K[0, 0] * (1 + zoom)
    return Kp, Rp, -Rp @ np.asarray(centre, float)


def test_register_lines_counts_through_a_gap_and_a_duplicate():
    K, R, t = camera()
    rows = ep.line_rows(K, R, t, 960.0)
    seen = [0, 1, 2, 4]                                    # line 3 hidden by players
    lines = [(rows[k], None, None) for k in seen]
    lines.insert(3, (rows[2] + 8.0, None, None))           # a second cluster of line 2
    ks = ep.register_lines(lines, K, R, t, goal_row=rows[0] - 25.0)
    assert ks == [0, 1, 2, 2, 4]


def test_register_lines_tracking_mode_uses_the_nearest_projected_line():
    K, R, t = camera()
    rows = ep.line_rows(K, R, t, 960.0)
    lines = [(rows[k] + 3.0, None, None) for k in (1, 2, 5)]
    assert ep.register_lines(lines, K, R, t, goal_row=None) == [1, 2, 5]


def test_column_inliers_rejects_strays():
    rng = np.random.default_rng(1)
    on = np.stack([600 - 0.1 * np.arange(0, 800, 20), np.arange(0, 800, 20)], 1)
    off = rng.uniform([300, 0], [900, 800], (8, 2))
    pts = np.concatenate([on, off])
    m = ep._column_inliers(pts, tol_px=3.0)
    assert m[:len(on)].all()
    assert m[len(on):].sum() <= 1


def test_fit_frame_recovers_a_perturbed_camera():
    from scipy.spatial.transform import Rotation

    centre = (88.0, 0.6, 21.0)
    K, R, t = camera(centre)
    pf = synthetic_frame(K, R, t, noise=0.5)
    Kp, Rp, tp = perturb(K, R, centre)
    assert ep.frame_score(Kp, Rp, tp, pf)[0] > 20
    res = ep.fit_frame(pf, Kp, Rp, centre)
    assert res.applied
    assert res.after_px < 1.5
    assert res.dash_px < 1.5
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(res.R @ R.T).as_rotvec())) < 0.05
    assert abs(res.K[0, 0] / K[0, 0] - 1) < 0.005


def test_fit_centre_recovers_the_mount():
    centre = np.array([88.0, 0.6, 21.0])
    pfs, K0s, R0s = [], [], []
    for target, f in (((-20.0, 0.0, 0.0), 15000.0), ((-24.0, 2.0, 0.0), 14000.0),
                      ((-16.0, -2.0, 0.0), 16000.0), ((-28.0, 1.0, 0.0), 13500.0)):
        K, R, t = camera(centre, target, f)
        pfs.append(synthetic_frame(K, R, t, noise=0.5))
        Kp, Rp, _ = perturb(K, R, centre, deg=0.8, zoom=0.05)
        K0s.append(Kp)
        R0s.append(Rp)
    c_hat, per = ep.fit_centre(pfs, K0s, R0s, centre + np.array([-20.0, 2.0, -4.0]))
    assert np.linalg.norm(c_hat - centre) < 2.0
    for pf, K0, R0, p in zip(pfs, K0s, R0s, per):
        K, R, t = ep.camera_from(p, K0, R0, c_hat)
        assert ep.frame_score(K, R, t, pf)[0] < 2.0


def test_red_green_boundary_on_a_synthetic_frame():
    img = np.zeros((300, 400, 3), np.uint8)
    img[:100] = (30, 40, 200)            # BGR: red end zone
    img[100:104] = (240, 240, 240)       # the white line at its edge
    img[104:] = (50, 150, 60)            # green turf
    b = ep.red_green_boundary(img, cols=(0, 400))
    assert b is not None and abs(b - 100) <= 6
    green = np.full((300, 400, 3), (50, 150, 60), np.uint8)
    assert ep.red_green_boundary(green, cols=(0, 400)) is None


def test_player_rulers_meet_on_a_synthetic_point():
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import CameraTrack

    Ka, Ra, ta = camera((-3.7, -101.6, 42.5), (-20.0, 0.0, 0.0), 9400.0)
    Kb, Rb, tb = camera()
    X = np.array([[-20.0, 1.0, 0.1], [-20.0, 1.3, 0.1], [-20.0, 1.0, 0.95], [-20.0, 1.3, 0.95],
                  [-22.0, -2.0, 0.1], [-22.0, -1.7, 0.1], [-22.0, -2.0, 0.95], [-22.0, -1.7, 0.95]])
    joints = [15, 16, 11, 12, 15, 16, 11, 12]
    pids = [1, 1, 1, 1, 2, 2, 2, 2]
    rows = []
    for cam, (K, R, t) in (("sideline", (Ka, Ra, ta)), ("endzone", (Kb, Rb, tb))):
        uv = project(K, R, t, X)
        for (u, v), j, pid in zip(uv, joints, pids):
            rows.append({"frame": 0, "cam": cam, "global_player_id": pid, "joint": j, "x": u, "y": v, "conf": 0.9})
    kdf = pd.DataFrame(rows)
    ta_ = CameraTrack(K=Ka[None], R=Ra[None], t=ta[None], conf=np.ones(1), width=1920, height=1080)
    tb_ = CameraTrack(K=Kb[None], R=Rb[None], t=tb[None], conf=np.ones(1), width=1920, height=1080)
    r = ep.player_rulers(ta_, tb_, kdf)
    assert len(r["frames"]) == 1
    assert r["miss_p50"][0] < 1e-6
    assert abs(r["ankle_z"][0] - 0.1) < 1e-6
    assert abs(r["hip_z"][0] - 0.95) < 1e-6
