import numpy as np


def test_detect_landmarks_maps_peaks_to_source_coords():
    from nfl_gsplat.landmarks.heatmap import render_gaussian
    from nfl_gsplat.landmarks.infer import detect_landmarks, landmarks_to_correspondences
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    K = s.num_classes
    hh, ww = 135, 240
    heat = np.zeros((K, hh, ww), np.float32)
    heat[0] = render_gaussian((hh, ww), (120.0, 67.0), 2.0)
    dets = detect_landmarks(heat, s, src_hw=(1080, 1920), in_hw=(540, 960),
                            heat_stride=4, conf_thresh=0.5)
    assert len(dets) == 1
    name, (u, v), conf = dets[0]
    assert name == s.class_names()[0] and conf > 0.9
    assert abs(u - 960.0) < 3 and abs(v - 536.0) < 3
    corrs = landmarks_to_correspondences(dets, s)
    assert corrs == [(name, (u, v))]
