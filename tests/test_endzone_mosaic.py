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


def test_register_to_reference_masks_players_via_boxes_by_frame(monkeypatch):
    """I1: register_to_reference must exclude players from feature detection
    (keep_mask's own docstring: 'excluded before feature detection') --
    before this fix, features were computed with NO mask at all and the
    function had no boxes parameter."""
    base = _textured_field()
    frames = {0: base, 1: _warp(base, np.eye(3))}
    boxes_by_frame = {0: [(50, 50, 150, 150)], 1: []}

    captured_masks = {}
    real_features = em._features

    def spy_features(img_bgr, mask=None):
        captured_masks[id(img_bgr)] = None if mask is None else mask.copy()
        return real_features(img_bgr, mask)
    monkeypatch.setattr(em, "_features", spy_features)

    em.register_to_reference(frames, ref_idx=0, boxes_by_frame=boxes_by_frame)

    mask0 = captured_masks[id(frames[0])]
    assert mask0 is not None, "boxes_by_frame must reach feature detection as a mask"
    assert mask0[100, 100] == 0            # inside the boxed player region
    assert mask0[10, 10] == 255            # untouched field


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


def test_accumulate_survives_player_parked_on_the_line_of_scrimmage():
    """C1 RED/GREEN: linemen stand still on the line of scrimmage, so a player
    box covering a painted line in the MAJORITY of frames must not erase that
    line. Before the fix, `seen` warped an all-ones image regardless of
    masking, so every frame claimed to have observed pixels it deliberately
    discarded via the player mask -- diluting the under-player vote by frames
    where paint could never have been counted there in the first place
    (measured: 0.40, below the 0.5 survival threshold, vs 1.00 once `seen`
    only counts player-masked-KEPT coverage)."""
    h, w = 300, 400
    frames, boxes, H_by = {}, {}, {}
    line_y = 150
    box = (180, 140, 220, 160)          # straddles the line at x in [180, 220]
    for i in range(10):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)   # green field (BGR)
        cv2.line(img, (0, line_y), (w, line_y), (250, 250, 250), 3)   # STATIC paint
        frames[i], H_by[i] = img, np.eye(3)
        boxes[i] = [box] if i < 6 else []      # player parked on the line 6/10 frames
    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    under_player_vote = acc[line_y, 200]        # inside the box's x-range, on the line
    assert under_player_vote >= 0.5, (
        f"line under a majority-frames-parked player should survive the "
        f"vote threshold, got {under_player_vote}")


def test_accumulate_floors_low_support_single_frame_artifact():
    """Item 2 RED/GREEN: the mirror image of C1. A pixel eligible (keep=1)
    in only 1 of 10 frames, where that one frame happens to show a bright
    non-paint artifact (glare/towel/jersey), must NOT be accepted as static
    paint just because votes/seen = 1/1 = 1.0 -- that ratio carries no
    statistical weight. A genuinely static line present in 6/10 frames (the
    C1 case) must still survive alongside it."""
    h, w = 300, 400
    frames, boxes, H_by = {}, {}, {}
    line_y = 150
    player_box = (180, 140, 220, 160)      # C1 case: line occluded 6/10 frames
    artifact_box = (280, 40, 320, 60)      # mirror case: occluded 9/10 frames
    artifact_px = (50, 300)                # (y, x) inside artifact_box

    for i in range(10):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)
        cv2.line(img, (0, line_y), (w, line_y), (250, 250, 250), 3)   # static paint
        boxes_i = []
        if i < 6:
            boxes_i.append(player_box)          # C1: line occluded 6/10 frames
        if i != 0:
            boxes_i.append(artifact_box)        # artifact region occluded 9/10 frames
        else:
            # the ONE frame where the artifact region is visible: a bright
            # non-paint artifact (glare/towel/jersey), not real paint
            cv2.rectangle(img, artifact_box[:2], artifact_box[2:], (250, 250, 250), -1)
        frames[i], boxes[i], H_by[i] = img, boxes_i, np.eye(3)

    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    line_vote = acc[line_y, 200]
    artifact_vote = acc[artifact_px]
    assert line_vote >= 0.5, f"C1 line must still survive, got {line_vote}"
    assert artifact_vote == 0.0, (
        f"single-frame artifact with 1/10 support must be floored to 0, "
        f"got {artifact_vote}")


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


def test_solve_reference_camera_recovers_a_known_planar_camera():
    """Project field points through a KNOWN camera, then solve it back."""
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ C
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()

    world, uv = [], []
    for X in (-18.288, -13.716, -9.144, -4.572, 0.0):
        for Y in (-20.0, 20.0):
            p = np.array([[X, Y, 0.0]])
            q = project_points(p, K, R, t)[0]
            if np.isfinite(q).all():
                world.append([X, Y, 0.0]); uv.append(q)
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(1500, 3500))
    cam = em.solve_reference_camera(world, uv, wh, prior)
    assert np.allclose(cam.pose.center_world(), C, atol=0.5)
    assert abs(cam.intrinsics.fx - 2600.0) / 2600.0 < 0.02
    assert cam.rms_px < 1.0


def test_solve_reference_camera_rejects_a_centre_outside_the_prior():
    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.errors import CalibrationError
    from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ C
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()
    world, uv = [], []
    for X in (-18.288, -13.716, -9.144, -4.572, 0.0):
        for Y in (-20.0, 20.0):
            q = project_points(np.array([[X, Y, 0.0]]), K, R, t)[0]
            if np.isfinite(q).all():
                world.append([X, Y, 0.0]); uv.append(q)
    wrong = EndzonePrior(x_range=(60, 150), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(1500, 3500))
    with pytest.raises(CalibrationError, match="prior box"):
        em.solve_reference_camera(world, uv, wh, wrong)


def test_propagate_sign_normalises_before_rq_when_det_m_is_negative():
    """H is only defined up to an arbitrary nonzero homogeneous scale, so
    `M = inv(H) @ M_ref` can come out with a negative determinant purely
    from that scale ambiguity -- not because the frame camera is a
    reflection. propagate() must normalise the sign BEFORE decomposing:
    multiplying K by diag(-1,-1,1) AFTER decomposition cannot fix a wrong
    det(R) (det(diag(-1,-1,1)) = +1) and instead corrupts K's focal signs."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    ref = CalibrationResult(
        intrinsics=CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.0, num_correspondences=0,
        refined_with_ba=False)

    s = 1.15
    cx, cy = wh[0] / 2, wh[1] / 2
    H_t = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])
    H_neg = -H_t          # the SAME homography -- homogeneous scale is arbitrary

    M_ref = ref.intrinsics.K() @ ref.pose.R
    M = np.linalg.inv(H_neg) @ M_ref
    assert np.linalg.det(M) < 0        # actually drives the sign-flip branch

    out = em.propagate({7: H_neg}, ref, n_frames=8)
    cam = out[7]
    assert cam is not None
    assert np.linalg.det(cam.pose.R) > 0

    M_signed = -M if np.linalg.det(M) < 0 else M
    M_rec = cam.intrinsics.K() @ cam.pose.R
    assert np.allclose(M_rec / M_rec[2, 2], M_signed / M_signed[2, 2])


def test_propagate_drops_a_frame_that_decomposes_anisotropic():
    """C3 RED/GREEN: whatever the RQ decomposition yields used to be emitted
    verbatim, with the reference camera's rms_px/num_correspondences copied
    onto it -- so a frame resolved via a weak/degenerate homography (e.g. the
    one-hop registration fallback) could come out grossly anisotropic
    (measured: fx=1300, fy=5200, a 4:1 K) yet still carry the reference's
    confidence numbers, giving a downstream consumer no signal it was
    garbage. An anisotropic H (unequal x/y scale about the principal point --
    not a pure zoom) must now be REJECTED (out[t] is None), and an accepted
    frame must never wear the reference's rms_px/num_correspondences."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    ref = CalibrationResult(
        intrinsics=CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.31, num_correspondences=18,
        refined_with_ba=False)

    cx, cy = wh[0] / 2, wh[1] / 2
    sx, sy = 1.15, 4.0          # anisotropic scale -- not a pure zoom
    H_aniso = np.array([[sx, 0, cx * (1 - sx)], [0, sy, cy * (1 - sy)], [0, 0, 1.0]])

    # Padded with several identity-homography frames (each trivially matches
    # the reference exactly, always accepted) so this single anisotropic drop
    # stays well under the item-3 "more than half the sampled frames dropped"
    # aggregate gate -- isolating the PER-FRAME gate under test here from that
    # separate aggregate check (covered on its own below).
    H_by = {i: np.eye(3) for i in range(6)}
    H_by[7] = H_aniso
    out = em.propagate(H_by, ref, n_frames=8)
    assert out[7] is None, "anisotropic frame must be dropped, not emitted"

    # A genuine isotropic zoom must still be accepted -- but never wearing
    # the reference's confidence numbers (they were never re-measured for
    # this frame).
    s = 1.15
    H_iso = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])
    out2 = em.propagate({7: H_iso}, ref, n_frames=8)
    cam = out2[7]
    assert cam is not None
    assert abs(cam.intrinsics.fx / cam.intrinsics.fy - 1.0) < 0.05
    assert cam.rms_px == 0.0
    assert cam.num_correspondences == 0


def _ref_cam_2600():
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return wh, CalibrationResult(
        intrinsics=CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.0, num_correspondences=0,
        refined_with_ba=False)


def test_propagate_no_focal_gate_when_focal_range_omitted():
    """Item 3: the removed +-20%-of-reference focal window was measured too
    narrow for real zoom spans (acceptance band s in [0.85, 1.25], barely
    1.5x total zoom). With focal_range=None (the default), only the aspect
    gate applies -- a big but ISOTROPIC zoom that lands fx far outside any
    +-20% window must still be accepted."""
    wh, ref = _ref_cam_2600()
    cx, cy = wh[0] / 2, wh[1] / 2
    s = 2.5                                    # fx ends up at 2600/2.5=1040
    H_big_zoom = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])

    out = em.propagate({7: H_big_zoom}, ref, n_frames=8)
    cam = out[7]
    assert cam is not None
    assert abs(cam.intrinsics.fx / cam.intrinsics.fy - 1.0) < 0.05
    assert abs(cam.intrinsics.fx / 2600.0 - 1.0) > 0.20, \
        "test must actually exercise fx outside the old +-20% window"


def test_propagate_focal_range_gate_when_provided():
    """When the driver passes the operator's own endzone_prior.focal_range,
    a frame whose fx falls outside it is dropped even though its aspect is
    perfectly isotropic."""
    wh, ref = _ref_cam_2600()
    cx, cy = wh[0] / 2, wh[1] / 2
    s = 2.5                                    # fx ends up at 1040, outside (1500,3500)
    H_big_zoom = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])

    # Padded with identity-homography frames -- see the note in
    # test_propagate_drops_a_frame_that_decomposes_anisotropic.
    H_by = {i: np.eye(3) for i in range(6)}
    H_by[7] = H_big_zoom
    out = em.propagate(H_by, ref, n_frames=8, focal_range=(1500.0, 3500.0))
    assert out[7] is None


def test_propagate_raises_when_more_than_half_sampled_frames_dropped():
    """Drops must never be silent: if more than half the sampled frames fail
    the sanity gate, propagate must raise loud naming the count, rather than
    handing assemble_track_from_results a mostly-empty track that silently
    interpolates or clamp-extrapolates over the gap."""
    import pytest

    from nfl_gsplat.errors import CalibrationError

    wh, ref = _ref_cam_2600()
    cx, cy = wh[0] / 2, wh[1] / 2
    H_bad = np.array([[1.15, 0, cx * (1 - 1.15)], [0, 4.0, cy * (1 - 4.0)], [0, 0, 1.0]])
    s = 1.15
    H_good = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])

    with pytest.raises(CalibrationError, match=r"3/4"):
        em.propagate({0: H_bad, 1: H_bad, 2: H_bad, 3: H_good}, ref, n_frames=4)


def test_solve_reference_camera_wraps_degenerate_focal_as_calibration_error(monkeypatch):
    """homography_to_krt (via _solve_focal) raises a bare ValueError on a
    degenerate view; solve_reference_camera must not let that escape -- the
    binding constraint is fail loud with CalibrationError + a pointer."""
    import pytest

    from nfl_gsplat.calibration import decompose_homography
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.errors import CalibrationError
    from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ C
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()

    world, uv = [], []
    for X in (-18.288, -13.716, -9.144, -4.572, 0.0):
        for Y in (-20.0, 20.0):
            q = project_points(np.array([[X, Y, 0.0]]), K, R, t)[0]
            if np.isfinite(q).all():
                world.append([X, Y, 0.0]); uv.append(q)
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(1500, 3500))

    def _boom(H, *, width, height):
        raise ValueError("cannot recover focal from homography (degenerate view)")

    monkeypatch.setattr(decompose_homography, "homography_to_krt", _boom)
    with pytest.raises(CalibrationError, match="degenerate"):
        em.solve_reference_camera(world, uv, wh, prior)


def _labelled_field_correspondences(wh, C, fwd_xz_only=True):
    """Shared helper: project the standard 10-point field grid through a
    known camera and return (world, uv)."""
    from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points

    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ C
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()
    world, uv = [], []
    for X in (-18.288, -13.716, -9.144, -4.572, 0.0):
        for Y in (-20.0, 20.0):
            q = project_points(np.array([[X, Y, 0.0]]), K, R, t)[0]
            if np.isfinite(q).all():
                world.append([X, Y, 0.0]); uv.append(q)
    return world, uv


def test_solve_reference_camera_rejects_high_rms_from_one_bad_line():
    """One yard line labelled a gridline off keeps its true pixel location
    but gets assigned the WRONG world X. RANSAC correctly excludes it from
    the homography fit (9/10 inliers, above the inlier-fraction gate), so
    only the rms gate catches it: reprojected against ALL 10 supplied
    points (including the one RANSAC excluded), rms comes out at ~11.9 px
    -- well above the 3.0 px gate tied to the RANSAC fit threshold used
    above. Measured previously: this labeling was silently ACCEPTED before
    this gate existed."""
    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.errors import CalibrationError

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    world, uv = _labelled_field_correspondences(wh, C)
    world[0] = [world[0][0] + 4.572, world[0][1], world[0][2]]   # one line off by 5 yd
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(1500, 3500))
    with pytest.raises(CalibrationError, match="rms"):
        em.solve_reference_camera(world, uv, wh, prior)


def test_solve_reference_camera_rejects_a_low_inlier_fraction():
    """4 of the 10 labelled points are off by 5-10 yards each: RANSAC keeps
    only 6/10 as inliers (60%), below the 70% clear-majority gate -- too
    much of the input disagrees with itself to trust whatever fit RANSAC
    happens to settle on, independent of what its rms would report."""
    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.errors import CalibrationError

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    world, uv = _labelled_field_correspondences(wh, C)
    for i in range(4):
        off_yd = (5 + 5 * (i % 2)) * (1 if i % 2 == 0 else -1)
        world[i] = [world[i][0] + off_yd * 0.9144, world[i][1], world[i][2]]
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(1500, 3500))
    with pytest.raises(CalibrationError, match="inlier"):
        em.solve_reference_camera(world, uv, wh, prior)
