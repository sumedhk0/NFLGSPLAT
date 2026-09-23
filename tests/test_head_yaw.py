import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.pose import head_yaw as hy


def _face(nose_x, lear_x, rear_x, conf=0.9, nose_conf=None, lear_conf=None, rear_conf=None):
    xy = np.full((17, 2), np.nan); c = np.zeros(17)
    xy[hy.NOSE] = [nose_x, 100]; xy[hy.L_EAR] = [lear_x, 100]; xy[hy.R_EAR] = [rear_x, 100]
    c[hy.NOSE] = conf if nose_conf is None else nose_conf
    c[hy.L_EAR] = conf if lear_conf is None else lear_conf
    c[hy.R_EAR] = conf if rear_conf is None else rear_conf
    return xy, c


def test_camera_yaw_from_the_nose_between_the_ears():
    # facing the camera: his left ear on the image's right, the nose centred
    yaw, cf = hy.head_yaw_camera(*_face(100, 120, 80))
    assert abs(yaw) < 1e-9 and cf == 0.9
    # nose toward his left ear (image right): a positive turn, a quarter turn when on the ear
    yaw, _ = hy.head_yaw_camera(*_face(120, 120, 80))
    assert abs(yaw - np.pi / 2) < 1e-9
    yaw, _ = hy.head_yaw_camera(*_face(90, 120, 80))
    assert -np.pi / 2 < yaw < 0
    # one ear hidden, the visible ear to the image-right of the nose: he faces image-left, past a quarter turn
    yaw, _ = hy.head_yaw_camera(*_face(100, 130, 0, rear_conf=0.1))
    assert yaw < -np.pi / 4
    yaw, _ = hy.head_yaw_camera(*_face(100, 0, 70, lear_conf=0.1))
    assert yaw > np.pi / 4
    # both ears, no nose: facing away
    yaw, _ = hy.head_yaw_camera(*_face(0, 120, 80, nose_conf=0.1))
    assert abs(yaw - np.pi) < 1e-9
    # nothing confident: no answer
    assert hy.head_yaw_camera(*_face(100, 120, 80, conf=0.2)) is None


def test_world_heading_uses_the_cameras_azimuth():
    # a camera looking along +x (world-to-camera R that maps world +x to camera +z)
    R = Rotation.from_euler("y", -90, degrees=True).as_matrix()      # camera z <- world x
    fwd = R.T @ np.array([0, 0, 1.0])
    assert abs(hy.camera_azimuth(R)) < 1e-9 and np.allclose(fwd, [1, 0, 0])
    # a man facing the camera looks along -x: heading pi
    assert abs(abs(hy.head_heading_world(0.0, R)) - np.pi) < 1e-9
    # facing away: heading 0 (along +x, with the camera)
    assert abs(hy.head_heading_world(np.pi, R)) < 1e-9
    # an UPRIGHT camera looking along +x (rows = camera right, down, forward in world): image-right is world -y
    # (fwd x up = x X z), so from facing the camera (heading pi) a quarter turn toward the image's right goes
    # counter-clockwise to -x-y: heading -3pi/4 -- the sign the old code had backwards (it was only ever tested
    # at 0 and pi, where the sign is invisible)
    R_up = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    assert abs(np.linalg.det(R_up) - 1) < 1e-9 and np.allclose(R_up.T @ [1.0, 0, 0], [0, -1, 0])
    assert abs(hy.camera_azimuth(R_up)) < 1e-9
    h = hy.head_heading_world(np.pi / 4, R_up)
    assert abs(h + 3 * np.pi / 4) < 1e-9
    # and a full profile toward image-right points straight along -y
    h = hy.head_heading_world(np.pi / 2, R_up)
    assert abs(h + np.pi / 2) < 1e-9


def test_head_headings_table():
    import pandas as pd
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    R = Rotation.from_euler("y", -90, degrees=True).as_matrix()
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    tr = CameraTrack(K=np.stack([K] * 3), R=np.stack([R] * 3), t=np.zeros((3, 3)), conf=np.ones(3), width=1920, height=1080)
    rows = []
    for f in range(3):
        for j in range(17):
            x = {hy.NOSE: 100, hy.L_EAR: 120, hy.R_EAR: 80}.get(j, 0)
            rows.append(dict(frame=f, cam="sideline", global_player_id=7, joint=j, x=x, y=100, conf=0.9 if j in (0, 3, 4) else 0.1))
    out = hy.head_headings(pd.DataFrame(rows), {"sideline": tr}, cam="sideline", frame_shift=0)
    assert set(out) == {7} and sorted(out[7]) == [0, 1, 2]
    heading, conf = out[7][1]
    assert abs(abs(heading) - np.pi) < 1e-9 and conf == 0.9
