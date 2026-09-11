"""Differentiable torch splatter over GaussianBatch (compositing.splat_torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nfl_gsplat.compositing import splat_torch as st  # noqa: E402
from nfl_gsplat.compositing.merge_ply import batch_from_arrays  # noqa: E402
from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at  # noqa: E402

W, H = 160, 120


def _camera():
    K = intrinsics(W, H, fov_deg=40.0)
    R, t = look_at(np.array([0.0, -8.0, 3.0]), np.array([0.0, 0.0, 1.0]))
    return K, R, t


def _batch(xyz, colours, sigma=0.15, opacity=0.95):
    n = len(xyz)
    rot = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    scale = np.full((n, 3), np.log(sigma), np.float32)
    opac = np.full(n, np.log(opacity / (1 - opacity)), np.float32)
    sh = ((np.asarray(colours, np.float32) - 0.5) / st.SH_C0)[:, :, None]
    return batch_from_arrays(np.asarray(xyz, np.float32), rot, scale, opac, sh)


def _pix(K, R, t, X):
    q = K @ (R @ np.asarray(X, float) + t)
    return q[:2] / q[2]


def test_projection_matches_the_pinhole_and_a_lone_gaussian_peaks_there():
    K, R, t = _camera()
    X = np.array([0.4, 0.2, 1.0])
    batch = _batch([X], [[1.0, 0.0, 0.0]])
    scene = st.SceneParams.from_batch(batch)
    means2d, cov2d, depth = st.project(scene, K, R, t)
    u, v = _pix(K, R, t, X)
    assert np.allclose(means2d[0].numpy(), [u, v], atol=1e-3)
    assert depth[0] > 0
    img = st.render(scene, K, R, t, crop=(0, 0, W, H), background=(0.0, 0.0, 0.0))
    r = img[..., 0].detach().numpy()
    pv, pu = np.unravel_index(np.argmax(r), r.shape)
    assert abs(pu - u) <= 1.0 and abs(pv - v) <= 1.0, (pu, pv, u, v)
    assert r.max() > 0.5 and img[..., 1].max() < 1e-6      # red only


def test_nearer_gaussian_occludes_the_farther_one():
    K, R, t = _camera()
    cam_centre = -R.T @ t
    X_far = np.array([0.0, 0.0, 1.0])
    X_near = cam_centre + 0.7 * (X_far - cam_centre)           # on the same ray
    batch = _batch([X_far, X_near], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], opacity=0.99)
    scene = st.SceneParams.from_batch(batch)
    img = st.render(scene, K, R, t, crop=(0, 0, W, H), background=(0.0, 0.0, 0.0))
    u, v = _pix(K, R, t, X_far)
    px = img[int(round(v)), int(round(u))].detach().numpy()
    assert px[0] > 0.9 and px[1] < 0.05, px                 # red wins in front


def test_gradients_reach_colour_scale_and_opacity():
    K, R, t = _camera()
    xyz = np.random.default_rng(0).normal(scale=0.3, size=(20, 3)) + [0, 0, 1]
    batch = _batch(xyz, np.random.default_rng(1).uniform(size=(20, 3)))
    scene = st.SceneParams.from_batch(batch, requires_grad=True)
    img = st.render(scene, K, R, t, crop=(40, 20, 80, 80))
    target = torch.zeros_like(img)
    loss = (img - target).abs().mean()
    loss.backward()
    for name in ("colour", "log_scale_mult", "opacity_logit"):
        g = getattr(scene, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_crop_offset_renders_the_same_pixels():
    K, R, t = _camera()
    batch = _batch([[0.2, 0.0, 1.0]], [[0.2, 0.6, 0.9]])
    scene = st.SceneParams.from_batch(batch)
    full = st.render(scene, K, R, t, crop=(0, 0, W, H))
    part = st.render(scene, K, R, t, crop=(50, 30, 40, 40))
    assert torch.allclose(full[30:70, 50:90], part, atol=1e-5)


def test_to_batch_round_trips_and_applies_the_fit():
    K, R, t = _camera()
    batch = _batch([[0.0, 0.0, 1.0], [0.3, 0.0, 1.2]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    scene = st.SceneParams.from_batch(batch)
    with torch.no_grad():
        scene.colour[0] = torch.tensor([0.9, 0.1, 0.1])
        scene.log_scale_mult[1] = np.log(2.0)
    out = scene.to_batch()
    assert np.allclose(out.xyz, batch.xyz)
    assert np.allclose(0.5 + st.SH_C0 * out.sh[0, :, 0], [0.9, 0.1, 0.1], atol=1e-6)
    assert np.allclose(np.exp(out.scale[1]), 2.0 * np.exp(batch.scale[1]), atol=1e-6)
    assert np.allclose(out.opacity, batch.opacity, atol=1e-6)


def test_a_dense_opaque_plane_renders_flat():
    # Millions of pairs per frame: the running log-transmittance sum reaches
    # tens of millions, where float32 resolves to ~4 and the per-pixel
    # difference turned into salt-and-pepper (every body, the whole field,
    # 2026-09-04). A dense uniform plane must come out uniform.
    K, R, t = _camera()
    n = 300_000
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.zeros(n)])
    batch = _batch(xyz, np.tile([0.8, 0.2, 0.1], (n, 1)), sigma=0.03, opacity=0.99)
    scene = st.SceneParams.from_batch(batch, device="cpu")
    img = st.render(scene, K, R, t, crop=(0, 0, W, H), background=(0.0, 0.0, 0.0))
    interior = img[80:115, 40:120]                            # the plane fills rows 70 and below
    assert interior.mean(dim=(0, 1))[0] > 0.7
    assert interior.std(dim=(0, 1)).max() < 0.02, interior.std(dim=(0, 1))

