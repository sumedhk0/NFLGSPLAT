"""Officials by their stripes (identity.officials)."""
import numpy as np

from nfl_gsplat.identity import officials as off


def _person(kind: str):
    """A 60x160 person crop on turf: striped shirt, plain jersey with a number, or plain."""
    import cv2

    img = np.full((160, 60, 3), (40, 110, 50), np.uint8)
    y0, y1 = int(0.18 * 160), int(0.55 * 160)
    if kind == "official":
        for x in range(4, 56, 6):
            img[y0:y1, x:x + 3] = (245, 245, 245)
            img[y0:y1, x + 3:x + 6] = (10, 10, 10)
    elif kind == "jersey":
        img[y0:y1, 4:56] = (30, 30, 200)
        cv2.putText(img, "52", (12, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    elif kind == "white":
        img[y0:y1, 4:56] = (240, 240, 240)
        cv2.putText(img, "8", (18, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    return img


def test_striped_shirt_scores_far_above_jerseys():
    box = (0, 0, 60, 160)
    s_off = off.stripe_score(_person("official"), box)
    s_jer = off.stripe_score(_person("jersey"), box)
    s_wht = off.stripe_score(_person("white"), box)
    assert s_off > 2.5 * max(s_jer, s_wht), (s_off, s_jer, s_wht)
    assert off.is_official([s_off] * 5, threshold=0.5 * (s_off + max(s_jer, s_wht)))
    assert not off.is_official([s_jer] * 5, threshold=0.5 * (s_off + max(s_jer, s_wht)))
    assert not off.is_official([s_off, s_off], threshold=0.1)     # too few crops


def test_tiny_box_is_no_measurement():
    assert np.isnan(off.stripe_score(np.zeros((160, 60, 3), np.uint8), (0, 0, 5, 5)))
