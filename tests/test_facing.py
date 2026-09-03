"""Facing score from the pose (identity.facing) and the OCR crop gate."""
import numpy as np
import pandas as pd

from nfl_gsplat.identity import facing


def test_square_to_the_lens_scores_one_sideways_zero():
    on_axis = [0.0, 0.0, 10.0]                        # body straight ahead of the camera
    assert abs(facing.facing_score([0.0, 0.0, 0.0], on_axis) - 1.0) < 1e-9      # back to lens
    assert abs(facing.facing_score([0.0, np.pi, 0.0], on_axis) - 1.0) < 1e-6    # chest to lens
    assert facing.facing_score([0.0, np.pi / 2, 0.0], on_axis) < 1e-6           # side-on


def test_off_axis_body_is_judged_along_its_own_ray():
    pos = np.array([6.0, 0.0, 8.0])                     # 37 deg off the optical axis
    # Turn the body so its forward points along the ray to the camera.
    ang = np.arctan2(pos[0], pos[2])
    assert abs(facing.facing_score([0.0, ang, 0.0], pos) - 1.0) < 1e-6
    assert facing.facing_score([0.0, ang + np.pi / 2, 0.0], pos) < 1e-6


def test_table_and_nearest_lookup():
    cache = {12: {3: {"global_orient": np.zeros(3), "joints3d_cam": np.array([[0, 0, 9.0]] * 5)}},
             18: {3: {"global_orient": np.array([0.0, np.pi / 2, 0.0]), "transl": np.array([0, 0, 9.0])}}}
    tab = facing.facing_table(cache)
    assert set(tab) == {(12, 3), (18, 3)}
    assert abs(tab[(12, 3)] - 1.0) < 1e-9 and tab[(18, 3)] < 1e-6
    assert facing.nearest(tab, 14, 3) == tab[(12, 3)]          # 2 away beats 4 away
    assert facing.nearest(tab, 16, 3) == tab[(18, 3)]
    assert np.isnan(facing.nearest(tab, 40, 3))
    assert np.isnan(facing.nearest(tab, 12, 9))


def test_ocr_crop_pick_prefers_facing_then_height():
    from nfl_gsplat.tracking.jersey_ocr import JerseyOCRConfig, _pick_rows

    g = pd.DataFrame({"frame": [0, 6, 12, 18], "bbox_y1": [0, 0, 0, 0],
                      "bbox_y2": [100, 200, 300, 400]})
    cfg = JerseyOCRConfig(top_k_frames=2, min_bbox_h_px=50, min_facing=0.5)
    # Tallest two are frames 18 and 12 -- but 18 is side-on and 12 unposed (NaN keeps).
    scores = {0: 0.9, 6: 0.8, 12: float("nan"), 18: 0.1}
    picked = _pick_rows(g, cfg, lambda frame: scores[int(frame)])
    assert list(picked["frame"]) == [12, 6]
    # Without a facing function the gate is off: pure height.
    assert list(_pick_rows(g, cfg, None)["frame"]) == [18, 12]
