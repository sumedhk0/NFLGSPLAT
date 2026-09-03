"""The number rows as a cross-field ruler (calibration.row_ruler)."""
import numpy as np
import pytest

from nfl_gsplat.calibration import row_ruler as rr

W, H = 1920, 1080


def _camera(f=9500.0, eye=(31.0, -100.0, 50.0), aim=(20.0, 0.0, 0.0)):
    from nfl_gsplat.compositing.preview_cpu import look_at

    K = np.array([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])
    R, t = look_at(np.array(eye, float), np.array(aim, float))
    return K, R, t


def _pix(K, R, t, X):
    q = (K @ (R @ np.asarray(X, float).T + t.reshape(3, 1))).T
    return q[:, :2] / q[:, 2:]


def test_camera_round_trips_through_its_homography():
    K, R, t = _camera()
    K2, R2, t2 = rr.camera_from_homography(rr.ground_homography(K, R, t), W, H)
    assert abs(K2[0, 0] - K[0, 0]) < 1e-6 * K[0, 0]
    assert np.allclose(R2, R, atol=1e-8) and np.allclose(t2, t, atol=1e-6)


def test_row_fit_from_both_rows_and_from_one():
    s_true, o_true = 1.3, -0.4
    ys = [(rr.ROW_Y_M - o_true) / s_true, (-rr.ROW_Y_M - o_true) / s_true]
    rs = rr.fit_row_scale(ys, [1, -1])
    assert abs(rs.scale - s_true) < 1e-9 and abs(rs.offset - o_true) < 1e-9
    one = rr.fit_row_scale([11.0, 11.2], [1, 1])
    assert one.offset == 0.0 and abs(one.scale - rr.ROW_Y_M / 11.1) < 0.02
    # Two wild readings out of eight (a frame with a poor camera) must not move it.
    far, near = ys
    wild = rr.fit_row_scale(ys + [far + 0.1, far - 0.1, near + 0.2, near - 0.1, 25.9, -7.0],
                            [1, -1, 1, 1, -1, -1, 1, -1])
    assert abs(wild.scale - s_true) < 0.03 and abs(wild.offset - o_true) < 0.3
    assert wild.residual_m < 0.5
    with pytest.raises(ValueError):
        rr.fit_row_scale([11.0], [1])


def test_refined_camera_moves_rows_and_keeps_yard_lines():
    """A solved camera whose world is squashed 0.78 in y (what play 1 had):
    build it by viewing a true camera through the squash, then refine it
    with the rows and check the true camera comes back."""
    K, R, t = _camera()
    s, o = 0.78, 0.3                       # solved y = s * true y + o
    H_true = rr.ground_homography(K, R, t)
    # Solved-world point (x, y) is true point (x, (y - o) / s).
    T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0 / s, -o / s], [0.0, 0.0, 1.0]])
    H_solved = H_true @ T
    Ks, Rs, ts = rr.camera_from_homography(H_solved, W, H)   # the wrong camera
    # Where the numerals would be read in the solved frame:
    rows_solved = [s * rr.ROW_Y_M + o, -s * rr.ROW_Y_M + o]
    rs = rr.fit_row_scale(rows_solved, [1, -1])
    K2, R2, t2 = rr.refine_camera(Ks, Rs, ts, rs, W, H)
    # The squashed world is not a camera's homography, so the "solved"
    # camera is a valid camera standing in for it -- which is exactly what
    # the paint solver hands over. The exact matrix path round-trips to
    # machine precision (checked in the diagnostic that set these numbers);
    # through the stand-in the refinement lands within twenty pixels (13.6
    # measured, at the frame's edge) where the stand-in was off by a
    # hundred or more at the rows.
    worst_before = worst_after = 0.0
    for x in (15.0, 20.0, 25.0):
        for y in (rr.ROW_Y_M, -rr.ROW_Y_M, 3.0):
            want = _pix(K, R, t, [[x, y, 0.0]])
            worst_before = max(worst_before, np.linalg.norm(_pix(Ks, Rs, ts, [[x, y, 0.0]]) - want))
            worst_after = max(worst_after, np.linalg.norm(_pix(K2, R2, t2, [[x, y, 0.0]]) - want))
    assert worst_after < 20.0, worst_after
    assert worst_before > 5 * worst_after, (worst_before, worst_after)
    assert np.linalg.norm((-R2.T @ t2) - np.array([31.0, -100.0, 50.0])) < 1.0
    assert abs(K2[0, 0] - K[0, 0]) < 0.02 * K[0, 0]
    assert (-R2.T @ t2)[2] > 0


def test_refine_track_applies_to_every_frame():
    from nfl_gsplat.calibration.cameras_io import CameraTrack

    K, R, t = _camera()
    track = CameraTrack(K=np.stack([K, K]), R=np.stack([R, R]), t=np.stack([t, t]),
                        conf=np.ones(2), width=W, height=H)
    rs = rr.RowScale(1.0, 0.0, 1, 1, 0.0)
    out = rr.refine_track(track, rs)
    assert np.allclose(out.R, track.R, atol=1e-8) and np.allclose(out.t, track.t, atol=1e-6)


def test_fit_rows_uses_every_known_row_and_reports_each_ruler():
    s_true, o_true = 1.06, 0.1

    def solved(yt):
        return (yt - o_true) / s_true

    ys = [solved(rr.ROW_Y_M), solved(-rr.ROW_Y_M), solved(rr.HASH_Y_M), solved(-rr.HASH_Y_M),
          solved(rr.HASH_Y_M) + 0.05]
    yt = [rr.ROW_Y_M, -rr.ROW_Y_M, rr.HASH_Y_M, -rr.HASH_Y_M, rr.HASH_Y_M]
    rulers = ["numerals", "numerals", "hashes", "hashes", "hashes"]
    fit = rr.fit_rows(ys, yt, rulers=rulers)
    assert abs(fit.scale - s_true) < 0.01 and abs(fit.offset - o_true) < 0.05
    assert set(fit.by_ruler) == {"numerals", "hashes"}
    assert abs(fit.by_ruler["numerals"] - s_true) < 1e-6
    assert abs(fit.by_ruler["hashes"] - s_true) < 0.03


def test_measure_hash_rows_finds_painted_ticks():
    """Paint hash ticks at +-HASH_Y_M on a synthetic turf seen by the camera."""
    import cv2

    K, R, t = _camera()
    frame = np.zeros((H, W, 3), np.uint8)
    Hg = rr.ground_homography(K, R, t)
    for x in np.arange(10.0, 30.0, 0.9144):                # every yard line
        for side in (1, -1):
            yc = side * rr.HASH_Y_M
            pts = np.array([[x - 0.1, yc - 0.3, 1], [x + 0.1, yc - 0.3, 1],
                            [x + 0.1, yc + 0.3, 1], [x - 0.1, yc + 0.3, 1]])
            q = pts @ Hg.T
            poly = (q[:, :2] / q[:, 2:]).astype(np.int32)
            cv2.fillPoly(frame, [poly], (255, 255, 255))
    rows = rr.measure_hash_rows(frame, K, R, t)
    got = {side: y for y, side in rows}
    assert set(got) == {1, -1}
    assert abs(got[1] - rr.HASH_Y_M) < 0.15 and abs(got[-1] + rr.HASH_Y_M) < 0.15, got

