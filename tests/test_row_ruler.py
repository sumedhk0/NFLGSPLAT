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
