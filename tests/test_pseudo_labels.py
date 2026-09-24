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
    assert S[s[9]] == "fit_self" and np.allclose(lab.uv["sideline"][9], proj["sideline"][9])   # anchored here: the fit's projection
    assert S[e[9]] == "fit_self" and np.allclose(lab.uv["endzone"][9], proj["endzone"][9])
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
    # a box entirely outside the image is no instance, nor is one whose labelled joints all fell outside it
    assert pl.yolo_pose_line((-50.0, -50.0, -10.0, -10.0), uv, vis, 200, 100) is None
    uv2 = np.full((17, 2), np.nan); vis2 = np.zeros(17, int); uv2[5], vis2[5] = (-3.0, 20.0), 2
    assert pl.yolo_pose_line((-20.0, 10.0, 60.0, 130.0), uv2, vis2, 200, 100) is None


def test_self_label_det_takes_the_detectors_point_only_on_request():
    """SELF_LABEL "det" (the detector's own point for a joint anchored in its camera) lost on film (2026-09-24); the
    default labels an anchored joint with the fit's two-camera projection, and "det" is there for an A/B."""
    proj = {"sideline": np.full((17, 2), np.nan)}
    proj["sideline"][13] = (100.0, 200.0)                       # the fit's left knee
    xy = np.full((17, 2), np.nan); conf = np.zeros(17)
    xy[13], conf[13] = (104.0, 197.0), 0.9                       # the detector's, 5 px away: anchored
    lab = pl.label_frame(proj, {"sideline": (xy, conf)}, agree_px=12.0, anchor_conf=0.5)
    assert pl.SOURCES[lab.source["sideline"][13]] == "fit_self" and lab.vis["sideline"][13] == 2
    assert np.allclose(lab.uv["sideline"][13], (100.0, 200.0))  # the fit's projection, SELF_LABEL "fit"
    lab_det = pl.label_frame(proj, {"sideline": (xy, conf)}, agree_px=12.0, anchor_conf=0.5, self_label="det")
    assert np.allclose(lab_det.uv["sideline"][13], (104.0, 197.0))  # the detector's point on request
    # the other camera's anchor still labels with the fit (the class the detector could not have produced)
    proj2 = {"sideline": proj["sideline"].copy(), "endzone": np.full((17, 2), np.nan)}
    proj2["endzone"][13] = (500.0, 300.0)
    xy_e = np.full((17, 2), np.nan); conf_e = np.zeros(17); xy_e[13], conf_e[13] = (503.0, 301.0), 0.9
    xy_s = np.full((17, 2), np.nan); conf_s = np.zeros(17)       # the sideline saw nothing
    lab2 = pl.label_frame(proj2, {"sideline": (xy_s, conf_s), "endzone": (xy_e, conf_e)}, agree_px=12.0, anchor_conf=0.5,
                          cross_joints=(13,))                     # a knee is cross-labelled only on request (CROSS_JOINTS)
    assert pl.SOURCES[lab2.source["sideline"][13]] == "fit_cross" and np.allclose(lab2.uv["sideline"][13], (100.0, 200.0))
    import pytest
    with pytest.raises(ValueError):
        pl.label_frame(proj, {"sideline": (xy, conf)}, self_label="other")


def test_cross_joints_restricts_the_cross_camera_labels_on_request():
    """By default the fit labels every body joint the other camera anchors (fit_cross); CROSS_JOINTS arms-only lost
    on film (2026-09-24) and stays an option: then a leg falls through to the detector's own point or nothing."""
    proj = {"sideline": np.full((17, 2), np.nan), "endzone": np.full((17, 2), np.nan)}
    for k in (10, 13):                                            # R wrist, L knee
        proj["sideline"][k] = (100.0 + k, 200.0); proj["endzone"][k] = (500.0 + k, 300.0)
    s_xy = np.full((17, 2), np.nan); s_conf = np.zeros(17)
    e_xy = np.full((17, 2), np.nan); e_conf = np.zeros(17)
    e_xy[10], e_conf[10] = proj["endzone"][10] + [2, 0], 0.9      # the endzone anchors both
    e_xy[13], e_conf[13] = proj["endzone"][13] + [2, 0], 0.9
    s_xy[13], s_conf[13] = proj["sideline"][13] + [30, 0], 0.6    # the sideline's own knee, far from the fit but confident
    lab = pl.label_frame(proj, {"sideline": (s_xy, s_conf), "endzone": (e_xy, e_conf)}, cross_joints=(5, 6, 7, 8, 9, 10))
    S = pl.SOURCES; src = lab.source["sideline"]
    assert S[src[10]] == "fit_cross" and np.allclose(lab.uv["sideline"][10], proj["sideline"][10])
    assert S[src[13]] == "det" and np.allclose(lab.uv["sideline"][13], s_xy[13])   # arms only: the knee is the detector's
    lab_all = pl.label_frame(proj, {"sideline": (s_xy, s_conf), "endzone": (e_xy, e_conf)})
    assert S[lab_all.source["sideline"][13]] == "fit_cross" and np.allclose(lab_all.uv["sideline"][13], proj["sideline"][13])  # the default: every joint
    lab_none = pl.label_frame(proj, {"sideline": (s_xy, s_conf), "endzone": (e_xy, e_conf)}, cross_joints=())
    assert S[lab_none.source["sideline"][10]] == "none" and lab_none.vis["sideline"][10] == 0   # no fit_cross at all
