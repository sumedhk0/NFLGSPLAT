"""Solve the endzone view in its own orientation, then map the camera back.

THE PROBLEM. Every line-labelling routine in this package assumes the sideline
camera's picture: yard lines run roughly UP the image, and the long lines it
must label -- the two sidelines and the two hash rows -- run ACROSS it. The
endzone camera looks down the length of the field, so the two families swap.
Given an endzone frame, the labeller takes yard lines for sidelines, and the
world it builds is stretched: measured on a production endzone clip, players
were placed across 101 m of a field that is 48.8 m wide.

THE FIX, AND WHY IT IS A ROTATION AND NOT A TRANSPOSE. Swapping u and v turns
one family into the other, which is why an unused ``_Transposed`` helper has sat
in from_paint for a while. But a transpose has determinant -1: it mirrors the
picture, so the camera that comes back out is left-handed, and the pipeline's
own mirror check correctly rejects it. Rotating by 90 degrees achieves the same
swap with determinant +1, so the recovered camera is a real camera, and the
rotation is absorbed exactly into R with no approximation:

    P_original = K' [R'R | R't]

with R' the 90 degree turn about the optical axis and K' the intrinsics with
their principal point rotated to match. Nothing is lost and nothing is fitted.
"""
from __future__ import annotations

import numpy as np

# A quarter turn about the optical axis, in the image plane.
_R90 = np.array([[0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])


def rotate_points_90(uv, width: int):
    """Where pixels land after the image is turned a quarter turn.

    Matches ``cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)``, which is what
    ``rotate_images_90`` applies -- the two must agree or the camera comes back
    reflected rather than rotated.
    """
    uv = np.asarray(uv, float).reshape(-1, 2)
    return np.column_stack([uv[:, 1], (width - 1) - uv[:, 0]])


def rotate_images_90(images):
    """``({frame: rotated}, width, height)`` -- the frames turned a quarter turn."""
    import cv2

    out = {f: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
           for f, img in images.items()}
    first = next(iter(out.values()))
    return out, int(first.shape[1]), int(first.shape[0])


def camera_from_rotated(K, R, t, orig_width: int):
    """Undo the quarter turn on a camera solved from rotated frames.

    The turn is a rotation of the image plane, so it composes with the camera's
    own rotation exactly; only the principal point has to be carried across.
    """
    K = np.asarray(K, float)
    f_x, f_y = K[0, 0], K[1, 1]
    c_u, c_v = K[0, 2], K[1, 2]
    # In the rotated picture u came from v, and v from the far side of u.
    K_out = np.array([[f_y, 0.0, (orig_width - 1) - c_v],
                      [0.0, f_x, c_u],
                      [0.0, 0.0, 1.0]])
    return K_out, _R90 @ np.asarray(R, float), _R90 @ np.asarray(t, float).reshape(3)


def cameras_from_rotated(cams, orig_width: int):
    return {f: camera_from_rotated(K, R, t, orig_width)
            for f, (K, R, t) in cams.items()}
