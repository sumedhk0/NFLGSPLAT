import numpy as np

from nfl_gsplat.pose import pseudo_labels as pl


def _det(xy, conf):
    return np.asarray(xy, float), np.asarray(conf, float)


def test_labels_follow_the_anchor_rule():
    proj = {"sideline": np.full((17, 2), np.nan), "endzone": np.full((17, 2), np.nan)}
    for k in pl.COCO_TO_SMPLX:
        proj["sideline"][k] = [100 + 10 * k, 200]
        proj["endzone"][k] = [500 + 10 * k, 300]
    side_xy = np.full((17, 2), np.nan); side_conf = np.zeros(17)
    end_xy = np.full((17, 2), np.nan); end_conf = np.zeros(17)
    # joint 9 (L wrist): the sideline saw it near the projection, the endzone too -> fit_self in both
    side_xy[9], side_conf[9] = proj["sideline"][9] + [3, 0], 0.9
    end_xy[9], end_conf[9] = proj["endzone"][9] + [0, 4], 0.8
    # joint 10 (R wrist): the sideline is unsure and far, the endzone anchors it -> fit_cross in the sideline
    side_xy[10], side_conf[10] = proj["sideline"][10] + [40, 0], 0.2
    end_xy[10], end_conf[10] = proj["endzone"][10] + [2, 2], 0.9
    # joint 7 (L elbow): the sideline is confident but 30 px off the projection, the endzone unsure -> the detector's own
    side_xy[7], side_conf[7] = proj["sideline"][7] + [30, 0], 0.8
    end_xy[7], end_conf[7] = proj["endzone"][7], 0.1
    # joint 8 (R elbow): nobody confident -> unlabelled
    side_xy[8], side_conf[8] = proj["sideline"][8], 0.3
    # face joint 0: the detector's own where confident
    side_xy[0], side_conf[0] = [50, 60], 0.95
    lab = pl.label_frame(proj, {"sideline": _det(side_xy, side_conf), "endzone": _det(end_xy, end_conf)})
    S = pl.SOURCES
    s = lab.source["sideline"]; e = lab.source["endzone"]
    assert S[s[9]] == "fit_self" and np.allclose(lab.uv["sideline"][9], proj["sideline"][9])
    assert S[e[9]] == "fit_self"
    assert S[s[10]] == "fit_cross" and np.allclose(lab.uv["sideline"][10], proj["sideline"][10]) and lab.vis["sideline"][10] == 2
    assert S[e[10]] == "fit_self"
    assert S[s[7]] == "det" and np.allclose(lab.uv["sideline"][7], side_xy[7])
    assert lab.vis["endzone"][7] == 0                                                # nothing anchors it and the endzone is unsure: unlabelled there
    assert lab.vis["sideline"][8] == 0 and lab.vis["endzone"][8] == 0
    assert S[s[0]] == "det" and np.allclose(lab.uv["sideline"][0], [50, 60]) and lab.vis["endzone"][0] == 0


def test_a_disagreeing_confident_detection_never_anchors():
    # the yaw trap: the sideline is confident but 30 px off the projection -> no anchor from it; the endzone (10 px
    # off, confident) anchors, so the sideline label is the projection (fit_cross), not the sideline detection
    proj = {"sideline": np.full((17, 2), np.nan), "endzone": np.full((17, 2), np.nan)}
    proj["sideline"][9] = [100, 100]; proj["endzone"][9] = [300, 300]
    lab = pl.label_frame(proj, {"sideline": _det(np.where(np.arange(17)[:, None] == 9, [130, 100], np.nan), np.where(np.arange(17) == 9, 0.9, 0)),
                                "endzone": _det(np.where(np.arange(17)[:, None] == 9, [300, 310], np.nan), np.where(np.arange(17) == 9, 0.9, 0))})
    assert pl.SOURCES[lab.source["sideline"][9]] == "fit_cross" and np.allclose(lab.uv["sideline"][9], [100, 100])
    lab2 = pl.label_frame(proj, {"sideline": _det(np.where(np.arange(17)[:, None] == 9, [130, 100], np.nan), np.where(np.arange(17) == 9, 0.9, 0))})
    assert pl.SOURCES[lab2.source["sideline"][9]] == "det"                       # no other camera: the detector's own


def test_yolo_line_and_pck():
    uv = np.full((17, 2), np.nan); vis = np.zeros(17, int)
    uv[5] = [960, 540]; vis[5] = 2
    line = pl.yolo_pose_line((900, 400, 1020, 700), uv, vis, 1920, 1080)
    parts = line.split()
    assert parts[0] == "0" and abs(float(parts[1]) - 0.5) < 1e-6 and abs(float(parts[3]) - 120 / 1920) < 1e-6
    assert parts[5 + 3 * 5:5 + 3 * 5 + 3] == ["0.500000", "0.500000", "2"] and parts[5:8] == ["0", "0", "0"]
    assert len(parts) == 5 + 51
    pred = np.full((17, 2), np.nan); pred[5] = [965, 545]
    assert pl.pck(pred, uv, vis, thr_px=10.0) == (1, 1) and pl.pck(pred, uv, vis, thr_px=5.0) == (0, 1)
    assert len(pl.COCO_FLIP) == 17 and all(pl.COCO_FLIP[pl.COCO_FLIP[k]] == k for k in range(17))


def test_coco_projection_puts_body_joints_at_their_coco_index():
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    R = np.eye(3); t = np.array([0.0, 0.0, 10.0])
    J = np.zeros((22, 3)); J[16] = [1.0, 0.0, 0.0]                                    # the left shoulder a metre to the right
    out = pl.coco_projection(J, K, R, t)
    assert np.isnan(out[0]).all() and np.allclose(out[5], [960 + 100, 540]) and np.allclose(out[11], [960, 540])


def test_detector_labels_take_confident_joints_or_leave_the_box_out():
    xy = np.arange(34, dtype=float).reshape(17, 2) + 100.0
    conf = np.zeros(17); conf[[5, 6, 11, 12, 15]] = 0.9
    u, v, s = pl.detector_labels(xy, conf, anchor_conf=0.5, min_joints=4)
    assert v.tolist() == [2 if k in (5, 6, 11, 12, 15) else 0 for k in range(17)]
    assert np.isnan(u[0]).all() and np.allclose(u[5], xy[5])
    assert all(pl.SOURCES[s[k]] == ("det" if v[k] else "none") for k in range(17))
    # a nan detection never qualifies; below min_joints the box is left out (None), never an all-unlabelled instance
    xy2 = xy.copy(); xy2[5] = np.nan
    assert pl.detector_labels(xy2, conf, anchor_conf=0.5, min_joints=5) is None
    assert pl.detector_labels(xy2, conf, anchor_conf=0.5, min_joints=4) is not None


def test_yolo_line_clips_the_box_and_drops_out_of_image_joints():
    uv = np.full((17, 2), np.nan); vis = np.zeros(17, int)
    uv[5], vis[5] = (10.0, 20.0), 2            # inside
    uv[6], vis[6] = (-3.0, 20.0), 2            # left of the image: unlabelled
    uv[7], vis[7] = (50.0, 105.0), 2           # below the image: unlabelled
    line = pl.yolo_pose_line((-20.0, 10.0, 60.0, 130.0), uv, vis, 200, 100)
    parts = line.split()
    cx, cy, w, h = (float(p) for p in parts[1:5])
    assert abs(cx - 30 / 200) < 1e-6 and abs(cy - 55 / 100) < 1e-6 and abs(w - 60 / 200) < 1e-6 and abs(h - 90 / 100) < 1e-6
    kp = np.array(parts[5:], float).reshape(17, 3)
    assert kp[5].tolist() == [10 / 200, 20 / 100, 2]
    assert kp[6].tolist() == [0, 0, 0] and kp[7].tolist() == [0, 0, 0]
    assert (kp[:, :2] >= 0).all() and (kp[:, :2] <= 1).all() and 0 <= cx - w / 2 and cx + w / 2 <= 1
    # a box entirely outside the image is no instance
    assert pl.yolo_pose_line((-50.0, -50.0, -10.0, -10.0), uv, vis, 200, 100) is None
