"""identity.torso_colours: the torso comes from the pose, because the box band is wrong in a stance."""
import numpy as np
import pandas as pd

from nfl_gsplat.identity.torso_colours import detection_colours, polygon_colour, torso_polygon


def test_the_polygon_is_the_shoulders_and_hips_pulled_in():
    j = {5: (10.0, 10.0), 6: (30.0, 10.0), 11: (12.0, 40.0), 12: (28.0, 40.0)}
    p = torso_polygon(j)
    assert p.shape == (4, 2)
    # inside the four corners, and centred on them
    assert np.allclose(p.mean(axis=0), np.array([20.0, 25.0]))
    assert p[:, 0].min() > 10.0 and p[:, 0].max() < 30.0
    assert torso_polygon({5: (1, 1), 6: (2, 2)}) is None       # a missing hip is no torso


def test_the_polygon_reads_the_jersey_not_the_pants():
    """A body in a stance: the jersey fills the TOP of the box and white pants the middle band."""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (255, 255, 255)                                  # white pants everywhere ...
    img[8:27, 20:80] = (0, 0, 255)                               # ... with a red jersey up top (BGR)
    joints = {5: (25.0, 11.0), 6: (75.0, 11.0), 11: (30.0, 24.0), 12: (70.0, 24.0)}
    poly = torso_polygon(joints)
    assert polygon_colour(img, poly)[1] > 200                    # saturated: the jersey
    df = pd.DataFrame([{"frame": 0, "track_id": 7, "bbox_x1": 10, "bbox_y1": 5, "bbox_x2": 90, "bbox_y2": 95}])

    class _Cap:                                                  # a one-frame stand-in for cv2.VideoCapture
        def set(self, *a):
            return True

        def read(self):
            return True, img

        def release(self):
            return None

    import nfl_gsplat.identity.torso_colours as tc
    import cv2
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *_a, **_k: _Cap()
    try:
        band = tc.detection_colours(df, "x.mp4")[0]
        pose = tc.detection_colours(df, "x.mp4", joints_by={(0, 7): joints})[0]
    finally:
        cv2.VideoCapture = orig
    assert band[1] < 60, band            # the band sits on the white pants
    assert pose[1] > 200, pose           # the pose finds the jersey


def test_a_tiny_torso_is_not_measured():
    img = np.zeros((20, 20, 3), np.uint8)
    poly = torso_polygon({5: (9.0, 9.0), 6: (11.0, 9.0), 11: (9.0, 11.0), 12: (11.0, 11.0)})
    assert not np.isfinite(polygon_colour(img, poly)).any()
