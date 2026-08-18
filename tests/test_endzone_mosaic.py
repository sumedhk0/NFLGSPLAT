import cv2
import numpy as np

from nfl_gsplat.calibration import endzone_mosaic as em


def _textured_field(w=640, h=480, seed=0):
    """Deterministic textured image with strong corners so SIFT has features."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, np.uint8)
    for i in range(9):                       # bright 'yard lines'
        y = 40 + i * 45
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 2)
    for _ in range(300):                     # texture for feature matching
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), 2, (200, 200, 200), -1)
    return img


def _warp(img, H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


def test_register_to_reference_recovers_known_homographies():
    base = _textured_field()
    truth = {0: np.eye(3)}
    frames = {0: base}
    for i, (dx, s) in enumerate([(12.0, 1.00), (25.0, 1.04), (-18.0, 0.97)], start=1):
        H = np.array([[s, 0.0, dx], [0.0, s, 0.5 * dx], [0.0, 0.0, 1.0]])
        truth[i] = H
        frames[i] = _warp(base, H)           # frame i = base warped by H

    H_by, inl = em.register_to_reference(frames, ref_idx=0)
    assert set(H_by) == {0, 1, 2, 3}
    assert np.allclose(H_by[0], np.eye(3), atol=1e-6)
    # H_by[i] maps frame i back INTO the reference, so it must invert truth[i]
    pts = np.float32([[100, 100], [500, 120], [300, 400]]).reshape(-1, 1, 2)
    for i in (1, 2, 3):
        back = cv2.perspectiveTransform(
            cv2.perspectiveTransform(pts, truth[i]), H_by[i])
        assert np.abs(back - pts).max() < 2.0, f"frame {i} round-trip off"
        assert inl[i] >= 25


def test_register_fails_loud_on_unregisterable_frame():
    import pytest

    from nfl_gsplat.errors import CalibrationError
    frames = {0: _textured_field(), 1: np.zeros((480, 640, 3), np.uint8)}
    with pytest.raises(CalibrationError, match="could not be registered"):
        em.register_to_reference(frames, ref_idx=0)


def test_keep_mask_zeroes_player_boxes_with_padding():
    m = em.keep_mask((100, 200, 3), [(50, 40, 70, 60)], pad=5)
    assert m[50, 60] == 0          # inside the box
    assert m[36, 46] == 0          # inside the pad
    assert m[10, 10] == 255        # untouched field


def test_register_fallback_bounded_to_one_hop_from_direct(monkeypatch):
    """Contiguous pending frames must each hop off a DIRECTLY-registered
    anchor, never off a frame that was itself only resolved by the fallback
    -- that is the naive sequential chaining the module explicitly rejects
    (measured drift 6px -> 282px on real footage)."""
    base = _textured_field()
    truth = {0: np.eye(3)}
    frames = {0: base}
    warps = [(12.0, 1.00), (25.0, 1.04), (38.0, 1.03)]
    for i, (dx, s) in enumerate(warps, start=1):
        H = np.array([[s, 0.0, dx], [0.0, s, 0.5 * dx], [0.0, 0.0, 1.0]])
        truth[i] = H
        frames[i] = _warp(base, H)
    # frame 0 = reference, frame 1 = the only frame that registers DIRECTLY,
    # frames 2 and 3 = a contiguous run with an (injected) weak direct link.

    owner_of = {id(img): i for i, img in frames.items()}
    real_features = em._features
    real_homography = em._homography
    feats_owner = {}
    calls = []  # (owner_a, owner_b, accepted)

    def tagging_features(img_bgr, mask=None):
        out = real_features(img_bgr, mask)
        feats_owner[id(out)] = owner_of[id(img_bgr)]
        return out

    def gated_homography(fa, fb, min_inliers):
        oa, ob = feats_owner.get(id(fa)), feats_owner.get(id(fb))
        if oa in (2, 3) and ob == 0:
            calls.append((oa, ob, False))
            return None, 0          # simulate a weak direct link to the ref
        H, n = real_homography(fa, fb, min_inliers)
        calls.append((oa, ob, H is not None and n >= min_inliers))
        return H, n

    monkeypatch.setattr(em, "_features", tagging_features)
    monkeypatch.setattr(em, "_homography", gated_homography)

    H_by, _ = em.register_to_reference(frames, ref_idx=0)

    assert set(H_by) == {0, 1, 2, 3}
    # The accepted fallback hop for frames 2 and 3 must anchor on frame 1 (the
    # only frame that registered DIRECTLY) -- never on each other. Anchoring
    # frame 3 on frame 2 would be exactly the rejected sequential chain.
    accepted_anchor = {oa: ob for oa, ob, ok in calls if ok and oa in (2, 3)}
    assert accepted_anchor[2] == 1
    assert accepted_anchor[3] == 1

    pts = np.float32([[100, 100], [500, 120], [300, 400]]).reshape(-1, 1, 2)
    for i in (1, 2, 3):
        back = cv2.perspectiveTransform(
            cv2.perspectiveTransform(pts, truth[i]), H_by[i])
        assert np.abs(back - pts).max() < 2.0, f"frame {i} round-trip off"


def test_accumulate_reinforces_static_paint_and_suppresses_movers():
    h, w = 300, 400
    frames, boxes, H_by = {}, {}, {}
    for i in range(8):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)   # green field (BGR)
        cv2.line(img, (0, 150), (w, 150), (250, 250, 250), 3)   # STATIC paint
        # a bright 'player' that moves every frame and is NOT boxed here
        cv2.rectangle(img, (20 + i * 40, 40), (60 + i * 40, 90), (255, 255, 255), -1)
        frames[i], boxes[i], H_by[i] = img, [], np.eye(3)
    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    line_vote = acc[150, w // 2]
    mover_vote = acc[65, 40]
    assert line_vote > 0.9, f"static paint should reinforce, got {line_vote}"
    assert mover_vote < 0.4, f"mover should wash out, got {mover_vote}"


def test_accumulate_recovers_line_occluded_in_some_frames():
    """A SkyCam cable breaks the line in a few frames; accumulation must heal it."""
    h, w = 200, 300
    frames, boxes, H_by = {}, {}, {}
    for i in range(10):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)
        cv2.line(img, (0, 100), (w, 100), (250, 250, 250), 3)
        if i < 3:                                   # cable occludes mid-span
            cv2.line(img, (140, 90), (140, 110), (20, 20, 20), 9)
        frames[i], boxes[i], H_by[i] = img, [], np.eye(3)
    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    assert acc[100, 140] > 0.6      # healed: unoccluded in 7/10 frames


def test_accumulate_masks_stationary_bright_box_via_boxes():
    """Coverage normalisation alone cannot suppress a STATIONARY bright box --
    it sits at the same pixels every frame and would otherwise rack up votes
    near 1.0, indistinguishable from paint. It must be suppressed ONLY
    because it is passed in boxes_by_frame and excluded by keep_mask. This
    guards the masking path itself: a swapped/inverted mask, a wrong shape
    argument, or a silently dropped boxes_by_frame parameter would let the
    box_vote assertion below fail even though the other accumulate tests
    (which pass empty boxes) would still pass."""
    h, w = 300, 400
    frames, boxes, H_by = {}, {}, {}
    box = (180, 40, 220, 90)                       # a player standing still
    for i in range(8):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)   # green field (BGR)
        cv2.line(img, (0, 150), (w, 150), (250, 250, 250), 3)   # STATIC paint
        cv2.rectangle(img, box[:2], box[2:], (255, 255, 255), -1)  # STATIONARY player
        frames[i] = img
        boxes[i] = [box]
        H_by[i] = np.eye(3)
    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    line_vote = acc[150, w // 2]
    box_vote = acc[65, 200]
    assert line_vote > 0.9, f"static paint should still reinforce, got {line_vote}"
    assert box_vote < 0.1, f"boxed stationary player should be masked out, got {box_vote}"


def test_propagate_matches_a_directly_rendered_camera():
    """Build a reference camera, warp by a known H, and check propagate()
    reproduces the camera that actually took the warped image."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()
    ref = CalibrationResult(
        intrinsics=CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.0, num_correspondences=0,
        refined_with_ba=False)

    # a pure zoom about the principal point: frame t -> reference
    s = 1.15
    cx, cy = wh[0] / 2, wh[1] / 2
    H_t = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])

    out = em.propagate({7: H_t}, ref, n_frames=8)
    cam = out[7]
    assert cam is not None
    # a world point must land where H_t would have put it
    X = np.array([[-30.0, 6.0, 0.0]])
    uv_ref = project_points(X, K, R, -R @ C)
    uv_t = project_points(X, cam.intrinsics.K(), cam.pose.R, cam.pose.t)
    back = cv2.perspectiveTransform(uv_t.reshape(-1, 1, 2).astype(np.float32), H_t)
    assert np.abs(back.reshape(-1, 2) - uv_ref).max() < 1.0
    assert np.allclose(cam.pose.center_world(), C, atol=1e-6)   # shared centre
