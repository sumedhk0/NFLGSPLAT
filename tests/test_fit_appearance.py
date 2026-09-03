"""Per-body photometric appearance fit (compositing.fit_appearance)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nfl_gsplat.compositing import fit_appearance as fa  # noqa: E402
from nfl_gsplat.compositing import splat_torch as st  # noqa: E402
from nfl_gsplat.compositing.mesh_to_gaussians import mesh_to_gaussians  # noqa: E402
from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at  # noqa: E402

W, H = 200, 160


def _sphere(n_lat=12, n_lon=18, r=0.5, centre=(0.0, 0.0, 1.0)):
    """A closed lat/long sphere mesh: vertices [V, 3], faces [F, 3]."""
    verts, faces = [], []
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            verts.append([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)])
    top, bottom = len(verts), len(verts) + 1
    verts += [[0, 0, r], [0, 0, -r]]
    idx = lambda i, j: (i - 1) * n_lon + (j % n_lon)  # noqa: E731
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            faces += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                      [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    for j in range(n_lon):
        faces += [[top, idx(1, j), idx(1, j + 1)], [bottom, idx(n_lat - 1, j + 1), idx(n_lat - 1, j)]]
    return np.asarray(verts) + np.asarray(centre), np.asarray(faces, np.int64)


def _cameras():
    K = intrinsics(W, H, fov_deg=35.0)
    out = []
    for eye in ([0.0, -5.0, 2.0], [4.0, -3.0, 1.5], [-4.0, -3.0, 2.5]):
        R, t = look_at(np.array(eye), np.array([0.0, 0.0, 1.0]))
        out.append((K, R, t))
    return out


def _truth_colours(verts):
    """Red where x > 0, blue where x < 0, smooth across the seam."""
    x = verts[:, 0]
    s = np.clip(0.5 + x / 0.3, 0, 1)[:, None]
    return s * np.array([[0.9, 0.1, 0.1]]) + (1 - s) * np.array([[0.1, 0.1, 0.9]])


def test_fit_recovers_vertex_colours_from_three_views():
    verts, faces = _sphere()
    truth = _truth_colours(verts)
    cams = _cameras()
    # Targets: the truth body rendered through each camera (full frames).
    obs = []
    for K, R, t in cams:
        batch = mesh_to_gaussians(verts, faces, colour=truth)
        scene = st.SceneParams.from_batch(batch)
        img = st.render(scene, K, R, t, crop=(0, 0, W, H), background=(0.2, 0.5, 0.2))
        obs.append(fa.FrameObs(image=img.detach().numpy(), K=K, R=R, t=t, vertices=verts))
    grey = np.full_like(truth, 0.5)
    fit, hist = fa.fit_body(grey, faces, obs, fa.FitConfig(iters=120, lr=0.05, tv_weight=0.002,
                                                          translation=False))
    assert hist["loss"][-1] < 0.4 * hist["loss"][0], hist["loss"][::20]
    seen = fa.seen_vertices(faces, obs)
    err = np.abs(fit.colour.detach().numpy() - truth)[seen].mean()
    assert err < 0.12, err
    assert np.abs(fit.colour.detach().numpy() - truth).max() <= 1.0


def test_translation_nuisance_absorbs_a_placement_error():
    """Targets rendered with the body 3 px off. With the nuisance on, the fit
    recovers the shift and leaves the (true) colours alone; without it, the
    only way down is to repaint the shift into the colours."""
    verts, faces = _sphere()
    truth = _truth_colours(verts)
    K, R, t = _cameras()[0]
    K_off = K.copy()
    K_off[0, 2] += 3.0
    batch = mesh_to_gaussians(verts, faces, colour=truth)
    scene = st.SceneParams.from_batch(batch)
    target = st.render(scene, K_off, R, t, crop=(0, 0, W, H), background=(0.2, 0.5, 0.2))
    ob = fa.FrameObs(image=target.detach().numpy(), K=K, R=R, t=t, vertices=verts)
    seen = fa.seen_vertices(faces, [ob])
    fit_on, hist_on = fa.fit_body(truth, faces, [ob], fa.FitConfig(iters=60, lr=0.05, tv_weight=0.0,
                                                                  translation=True, lr_shift=0.3))
    fit_off, _ = fa.fit_body(truth, faces, [ob], fa.FitConfig(iters=60, lr=0.05, tv_weight=0.0,
                                                             translation=False))
    assert abs(hist_on["shift"][0][0] - 3.0) < 1.0, hist_on["shift"]
    drift_on = np.abs(fit_on.colour.numpy() - truth)[seen].mean()
    drift_off = np.abs(fit_off.colour.numpy() - truth)[seen].mean()
    # Measured 0.023 against 0.040: Adam moves colours and the shift together
    # for the first iterations, so the nuisance halves the drift, not zeroes it.
    assert drift_on < 0.7 * drift_off, (drift_on, drift_off)


def test_seen_vertices_and_crop_are_sane():
    verts, faces = _sphere()
    K, R, t = _cameras()[0]
    ob = fa.FrameObs(image=np.zeros((H, W, 3), np.float32), K=K, R=R, t=t, vertices=verts)
    x0, y0, w, h = fa.crop_for(ob, margin_px=4)
    assert 0 <= x0 < x0 + w <= W and 0 <= y0 < y0 + h <= H and w > 10 and h > 10
    seen = fa.seen_vertices(faces, [ob])
    assert 0.3 < seen.mean() < 0.8            # one camera sees about half a sphere
