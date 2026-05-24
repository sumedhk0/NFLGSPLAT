"""End-to-end smoke test over the synthetic fixture.

Runs on CPU, no GPU / no external weights. Exercises the same call graph the
production pipeline uses, with mock LHM avatars and a mock field PLY standing
in for the GPU stages:

    fixture generate
        → solve_pnp (both cameras)         rms < 1 px
        → synthetic SMPLest-X projections  (world joints → (uv, conf) per cam)
        → triangulate_joints_two_view      < 5 cm
        → fuse_sequence (rigid forward)    recovers per-frame translation
        → mock field PLY + 3 mock avatars + mock ball
        → merge_batches                    count = field_N + 3·avatar_N + 1·ball_N
        → animate_gaussians                (LBS plumbing for one avatar)

The optional render assertion ("mean luminance > 5") lives in the gpu-marked
test further down and is skipped unless torch+gsplat import successfully.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nfl_gsplat.avatars.lbs_animate import animate_gaussians
from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.avatars.lhm_wrapper import resolve_avatars, write_mock_avatar
from nfl_gsplat.ball.ball_asset import make_football_asset
from nfl_gsplat.compositing.merge_ply import (
    batch_from_arrays,
    load_gaussian_ply,
    merge_batches,
    save_gaussian_ply,
)
from nfl_gsplat.compositing.scene import compose_frame, football_batch, posed_avatar_batch
from nfl_gsplat.identity.registry import EntityType
from nfl_gsplat.pose.fuse_smplx import resolve_betas
from nfl_gsplat.field.train_field import read_ply_gaussian_count, write_mock_field_ply
from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_annotations
from nfl_gsplat.pose.fuse_smplx import (
    SMPLXFitConfig,
    _pack_params,
    fuse_sequence,
    rigid_translation_forward,
)
from nfl_gsplat.pose.triangulate import TriangulationConfig, triangulate_joints_two_view
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points
from nfl_gsplat.utils.io import read_json
from tests.fixtures.generate import (
    FIXTURE_HEIGHT,
    FIXTURE_WIDTH,
    PLAYER_ROOTS,
    TEMPLATE_JOINTS_22,
    generate,
)


pytestmark = pytest.mark.smoke


# --- Fixture setup ----------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("nfl_smoke")
    generate(out)
    return out


def _cam_from_gt(gt: dict, width: int, height: int) -> tuple[CameraIntrinsics, CameraPose]:
    K = np.asarray(gt["K"])
    R = np.asarray(gt["R"])
    t = np.asarray(gt["t"])
    intr = CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=width, height=height,
    )
    return intr, CameraPose(R=R, t=t)


# --- 1. Calibration gate ----------------------------------------------------

def test_smoke_calibration_both_cameras(smoke_dir: Path):
    for cam in ("sideline", "endzone"):
        res = solve_pnp_from_annotations(
            smoke_dir / f"{cam}_landmarks.json",
            image_size=(FIXTURE_WIDTH, FIXTURE_HEIGHT),
            max_reproj_px=1.0,
            bundle_adjustment=True,
        )
        assert res.rms_px < 1.0, f"{cam} rms {res.rms_px:.3f} px exceeds smoke budget"


# --- 2. Triangulation gate --------------------------------------------------

def _synthesize_per_cam_observations(
    joints_world: np.ndarray,       # [T, J, 3]
    cam: tuple[CameraIntrinsics, CameraPose],
    noise_px: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Project world joints through a camera and fake a SMPLest-X output."""
    intr, pose = cam
    T, J, _ = joints_world.shape
    rng = np.random.default_rng(seed)
    flat = joints_world.reshape(-1, 3)
    uv = project_points(flat, intr.K(), pose.R, pose.t).reshape(T, J, 2)
    uv = uv + rng.normal(0.0, noise_px, uv.shape)
    conf = np.full((T, J), 0.95, dtype=np.float64)
    return {"uv": uv, "conf": conf}


def test_smoke_triangulation_recovers_joints(smoke_dir: Path):
    cams_gt = read_json(smoke_dir / "cameras_gt.json")
    intr_s, pose_s = _cam_from_gt(cams_gt["sideline"], FIXTURE_WIDTH, FIXTURE_HEIGHT)
    intr_e, pose_e = _cam_from_gt(cams_gt["endzone"], FIXTURE_WIDTH, FIXTURE_HEIGHT)

    # 3 players, 1 frame of T-pose joints each, stacked along T for the batch
    # API. joints_world[player_idx] = [1, 22, 3].
    joints_world = PLAYER_ROOTS[:, None, :] + TEMPLATE_JOINTS_22[None, :, :]  # [P, J, 3]
    joints_world = joints_world[:, None, :, :]                                 # [P, T=1, J, 3]

    for p in range(joints_world.shape[0]):
        obs = {
            "sideline": _synthesize_per_cam_observations(
                joints_world[p], (intr_s, pose_s), seed=10 + p),
            "endzone":  _synthesize_per_cam_observations(
                joints_world[p], (intr_e, pose_e), seed=20 + p),
        }
        tri = triangulate_joints_two_view(
            obs,
            {"sideline": (intr_s, pose_s), "endzone": (intr_e, pose_e)},
            TriangulationConfig(reproj_px_max=20.0, conf_min=0.3),
        )
        assert tri.valid.all(), f"player {p}: {int((~tri.valid).sum())} joints rejected"
        err_m = np.linalg.norm(tri.joints3d - joints_world[p], axis=-1)   # [T, J]
        assert err_m.max() < 0.05, (
            f"player {p}: max joint error {err_m.max()*100:.2f} cm exceeds 5 cm budget"
        )


# --- 3. SMPL-X fusion plumbing ---------------------------------------------

def test_smoke_fuse_recovers_translation(smoke_dir: Path):
    """Run the fusion optimizer over the triangulated joints for one player.

    Uses the rigid-translation forward (no body model) so the only solvable
    degree of freedom is transl — but that still exercises scipy, the masking
    logic, the warm-start loop, and the PoseFusionError gate.
    """
    cams_gt = read_json(smoke_dir / "cameras_gt.json")
    intr_s, pose_s = _cam_from_gt(cams_gt["sideline"], FIXTURE_WIDTH, FIXTURE_HEIGHT)
    intr_e, pose_e = _cam_from_gt(cams_gt["endzone"], FIXTURE_WIDTH, FIXTURE_HEIGHT)

    # Single player, 3 frames of translations.
    T = 3
    root_path = PLAYER_ROOTS[0:1] + np.array([[0, 0, 0], [1.0, 0.5, 0], [2.0, 1.0, 0]])
    joints_world = root_path[:, None, :] + TEMPLATE_JOINTS_22[None, :, :]     # [T, J, 3]

    obs = {
        "sideline": _synthesize_per_cam_observations(joints_world, (intr_s, pose_s), seed=100),
        "endzone":  _synthesize_per_cam_observations(joints_world, (intr_e, pose_e), seed=200),
    }
    tri = triangulate_joints_two_view(
        obs,
        {"sideline": (intr_s, pose_s), "endzone": (intr_e, pose_e)},
        TriangulationConfig(reproj_px_max=20.0, conf_min=0.3),
    )
    assert tri.valid.sum() >= T * 20

    cfg = SMPLXFitConfig(min_frame_validity_frac=0.6)
    template = PLAYER_ROOTS[0] + TEMPLATE_JOINTS_22        # [J, 3]
    forward = rigid_translation_forward(template, cfg)
    init = _pack_params(
        body_pose=np.zeros(cfg.body_pose_dim),
        global_orient=np.zeros(cfg.global_orient_dim),
        transl=np.zeros(cfg.transl_dim),
    )
    fit = fuse_sequence(tri.joints3d, tri.valid, init, forward, cfg)
    assert fit.valid_frames.all(), f"frames failed: {(~fit.valid_frames).nonzero()[0].tolist()}"
    gt_transl = root_path - PLAYER_ROOTS[0]
    err = np.linalg.norm(fit.transl - gt_transl, axis=-1)
    assert err.max() < 0.02, f"transl error {err.max()*100:.2f} cm exceeds 2 cm budget"


# --- 4. Composite count contract ------------------------------------------

def test_smoke_composite_count_matches_plan(tmp_path: Path):
    """Plan §9: composite PLY count == ``field_N + 3·avatar_N + 1·ball_N``."""
    field_path = tmp_path / "field.ply"
    avatar_path = tmp_path / "avatar.npz"
    write_mock_field_ply(field_path, num_gaussians=60_000, seed=1)
    write_mock_avatar(avatar_path, num_gaussians=3_000, num_joints=22, seed=2)
    assert read_ply_gaussian_count(field_path) == 60_000

    field_batch = load_gaussian_ply(field_path)

    # Load the mock avatar NPZ as a GaussianBatch (canonical pose — no LBS yet;
    # smoke is about count plumbing, not rigging).
    from nfl_gsplat.utils.io import read_npz
    av = read_npz(avatar_path)
    avatar_batch = batch_from_arrays(
        xyz=av["canonical_xyz"],
        rot=av["canonical_rot"],
        scale=av["canonical_scale"],
        opacity=av["canonical_opacity"],
        sh=av["canonical_sh"],
    )

    # Ball: small fixed-size Gaussian blob, sh_degree 0.
    ball_N = 500
    rng = np.random.default_rng(3)
    ball_batch = batch_from_arrays(
        xyz=rng.normal(0.0, 0.05, (ball_N, 3)).astype(np.float32),
        rot=np.tile([1, 0, 0, 0], (ball_N, 1)).astype(np.float32),
        scale=np.full((ball_N, 3), np.log(0.03), dtype=np.float32),
        opacity=np.full((ball_N,), 2.0, dtype=np.float32),
        sh=rng.normal(0.0, 0.1, (ball_N, 3, 1)).astype(np.float32),
    )

    merged = merge_batches([field_batch, avatar_batch, avatar_batch, avatar_batch, ball_batch])
    assert merged.num_gaussians == 60_000 + 3 * 3_000 + 500
    # Round-trip the merged PLY; count survives on disk.
    merged_ply = tmp_path / "merged.ply"
    save_gaussian_ply(merged_ply, merged)
    assert read_ply_gaussian_count(merged_ply) == merged.num_gaussians


# --- 5. LBS plumbing --------------------------------------------------------

def test_smoke_lbs_animation_moves_avatar(tmp_path: Path):
    """Identity + pure translation joint transforms shift canonical → world."""
    avatar_path = tmp_path / "av.npz"
    write_mock_avatar(avatar_path, num_gaussians=500, num_joints=22, seed=4)
    from nfl_gsplat.utils.io import read_npz
    av = read_npz(avatar_path)

    J = 22
    offset = np.array([5.0, -2.0, 0.0])
    tfms = np.tile(np.eye(4)[None, :, :], (J, 1, 1))
    tfms[:, :3, 3] = offset

    xyz_before = av["canonical_xyz"]
    xyz_after, _ = animate_gaussians(xyz_before, av["canonical_rot"], av["lbs_weights"], tfms)
    delta = xyz_after - xyz_before
    assert delta.shape == xyz_before.shape
    assert np.allclose(delta - offset[None, :], 0.0, atol=1e-6), (
        f"max delta drift {np.abs(delta - offset).max():.2e}"
    )


# --- 6. Optional render assertion (GPU only) -------------------------------

@pytest.mark.gpu
def test_smoke_render_nonblack(tmp_path: Path):
    """Render a single frame of the composite and assert mean luminance > 5.

    Skips if torch/gsplat is not importable, which is normal on CPU CI.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")

    from nfl_gsplat.compositing.render_gsplat import RenderConfig, render_trajectory

    if not torch.cuda.is_available():
        pytest.skip("gsplat requires CUDA")

    field_path = tmp_path / "field.ply"
    write_mock_field_ply(field_path, num_gaussians=60_000, seed=5)
    batch = load_gaussian_ply(field_path)

    intr = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    R, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    # Simple overhead-ish camera at 40 m altitude pointing down-forward.
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    t = np.array([0.0, 40.0, 30.0], dtype=np.float64)
    pose = CameraPose(R=R, t=t)

    frames_dir = tmp_path / "render"
    render_trajectory(batch, intr, poses=[pose], out_dir=frames_dir,
                      cfg=RenderConfig(), device="cuda:0")
    import imageio.v3 as iio
    pngs = sorted(frames_dir.glob("*.png"))
    assert len(pngs) == 1
    img = iio.imread(pngs[0]).astype(np.float32)
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    assert float(lum.mean()) > 5.0, f"rendered frame too dark: mean luminance {lum.mean():.2f}"


# --- 7. Season-scale path: library reuse + players/referee/football ----------

def test_smoke_composite_count_players_referee_football(tmp_path: Path):
    """Updated contract: field_N + Σ player_N + ref_N + football_N.

    Composes 2 posed players + 1 posed referee + 1 oriented football on top of
    the field, exactly as ``scripts/05`` does per frame.
    """
    field_path = tmp_path / "field.ply"
    write_mock_field_ply(field_path, num_gaussians=40_000, seed=1)
    field = load_gaussian_ply(field_path)
    field_n = field.num_gaussians

    lib = AvatarLibrary(tmp_path / "library", season=2024)
    from nfl_gsplat.utils.io import read_npz

    # Two players + one generic referee, each a mock canonical avatar.
    player_n, ref_n = 1500, 900
    for i, uid in enumerate(("00-A", "00-B")):
        ap = tmp_path / f"{uid}.npz"
        write_mock_avatar(ap, num_gaussians=player_n, num_joints=22, seed=10 + i)
        lib.put_avatar(uid, read_npz(ap))
    rp = tmp_path / "ref.npz"
    write_mock_avatar(rp, num_gaussians=ref_n, num_joints=22, seed=99)
    lib.put_referee_avatar(read_npz(rp))

    # Identity joint transforms → posed == canonical.
    tfms = np.tile(np.eye(4)[None, :, :], (22, 1, 1))
    posed = [posed_avatar_batch(lib.get_avatar(u), tfms) for u in ("00-A", "00-B")]
    posed.append(posed_avatar_batch(lib.get_referee_avatar(), tfms))

    asset = make_football_asset()
    ball_n = asset["xyz"].shape[0]
    ball = football_batch(asset, np.array([0.0, 0.0, 2.0]), np.array([8.0, 0.0, 1.0]), t=0.1)

    merged = compose_frame(field, posed, ball)
    assert merged.num_gaussians == field_n + 2 * player_n + ref_n + ball_n


def test_smoke_library_reuse_skips_regeneration(tmp_path: Path):
    """A player reconstructed in play 1 is served from the library in play 2 —
    LHM++ (mocked) runs once per player, not once per player-per-play."""
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    calls: list[str] = []

    def gen(uid: str) -> dict:
        calls.append(uid)
        ap = tmp_path / f"gen_{uid}.npz"
        write_mock_avatar(ap, num_gaussians=300, num_joints=22, seed=len(calls))
        from nfl_gsplat.utils.io import read_npz
        return read_npz(ap)

    P = EntityType.PLAYER.value
    resolve_avatars([("qb_12", P), ("wr_81", P)], lib, gen)
    plan2 = resolve_avatars([("qb_12", P), ("rb_28", P)], lib, gen)

    assert plan2.cache_hits == ["qb_12"]
    assert plan2.generated == ["rb_28"]
    assert calls == ["qb_12", "wr_81", "rb_28"], "qb_12 must not be regenerated"


def test_smoke_frozen_betas_reused_across_plays(tmp_path: Path):
    """Betas cached in play 1 are reused (not re-estimated) in play 2."""
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    betas = np.linspace(-0.2, 0.2, 10).astype(np.float32)
    lib.put_betas("qb_12", betas)

    cfg = SMPLXFitConfig(use_library_betas=True)
    # In a later play, the estimate would differ; the library value wins.
    resolved, source = resolve_betas(
        lib.get_betas("qb_12"), lambda: np.zeros(10), cfg
    )
    assert source == "library"
    assert np.allclose(resolved, betas, atol=1e-6)
