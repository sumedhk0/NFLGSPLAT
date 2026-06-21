from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.utils.geometry import project_points, triangulate_two_views


def _two_cam_tracks(T=3):
    def cam(yaw0, dyaw, cx_off):
        Ks, Rs, ts = [], [], []
        for i in range(T):
            y = np.deg2rad(yaw0 + dyaw * i)
            R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
            K = np.array([[2000.0, 0, 960], [0, 2000.0, 540], [0, 0, 1]], float)
            Ks.append(K); Rs.append(R); ts.append(np.array([cx_off, 5.0, 30.0]))
        return CameraTrack(np.array(Ks), np.array(Rs), np.array(ts), np.ones(T), 1920, 1080)
    return {"a": cam(10, 2.0, -20.0), "b": cam(-10, -2.0, 20.0)}


def test_per_frame_triangulation_recovers_moving_point():
    tracks = _two_cam_tracks()
    truth = [np.array([1.0 * i, 2.0, 0.5]) for i in range(3)]
    for frame, X in enumerate(truth):
        ia, pa = tracks["a"].at(frame)
        ib, pb = tracks["b"].at(frame)
        uva = project_points(X[None], ia.K(), pa.R, pa.t)
        uvb = project_points(X[None], ib.K(), pb.R, pb.t)
        Pa = ia.K() @ pa.extrinsic_3x4()
        Pb = ib.K() @ pb.extrinsic_3x4()
        Xhat = triangulate_two_views(uva, uvb, Pa, Pb)[0]
        assert np.allclose(Xhat, X, atol=1e-3)
