"""Body joints from fitted SMPL-X parameters, in the order the rest of the code assumes.

WHY THIS EXISTS. SMPLest-X returns ``joints3d_cam`` with 137 joints -- body,
hands and face in ITS OWN layout -- and the first 22 of those are NOT the
SMPL-X body tree that ``SMPLX_BODY_PARENTS`` describes. Treating them as if
they were connects unrelated joints, and every downstream length is then
meaningless. Measured on real output, taking ``joints3d_cam[:22]`` as the body
gave these "bones":

    l_elbow -> l_wrist    1.256 m
    r_collar -> r_shoulder + spine2 -> spine3    1.121 m and 0.918 m
    neck -> head          0.741 m

against a whole skeleton spanning 1.03 m. No human has a 1.26 m forearm. This
is the "tangled skeleton" the project had noticed and not pinned down; the
plausibility audit rejected 100% of real frames because of it.

WHAT TO DO INSTEAD. The fit returns betas, body_pose, global_orient and transl.
Pushing those back through the SMPL-X model returns joints in SMPL-X's own
order by construction, so the tree is right by definition rather than by
assumption. It costs one forward pass and removes a whole class of silent error.

Bone lengths still vary slightly frame to frame, because SMPLest-X re-estimates
betas every frame. That variation is real fit noise and is exactly what the
plausibility audit should be measuring -- unlike the mislabelled tree, which
was measuring nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.pose.forward_kinematics import NUM_BODY_JOINTS
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

DEFAULT_MODEL_DIR = Path("data/body_models")


def _load_model(model_dir, gender: str, batch: int, device: str):
    try:
        import smplx
    except ImportError as exc:  # pragma: no cover - env-gated
        raise SetupError(
            "smplx is not installed in this environment. The pose stack lives "
            "in C:/venvs/smplx312; see the local-machine notes."
        ) from exc

    model_dir = Path(model_dir)
    if not (model_dir / "smplx").exists():
        raise SetupError(
            f"SMPL-X body models not found under {model_dir}/smplx. "
            "Expected SMPLX_NEUTRAL.npz and friends.")
    return smplx.create(
        str(model_dir), model_type="smplx", gender=gender,
        use_pca=False, flat_hand_mean=True, batch_size=batch,
    ).to(device)


def body_joints(betas, body_pose, global_orient=None, transl=None, *,
                model_dir=DEFAULT_MODEL_DIR, gender: str = "neutral",
                device: str | None = None, batch: int = 64) -> np.ndarray:
    """``[N, 22, 3]`` SMPL-X body joints from fitted parameters.

    ``body_pose`` is ``[N, 21, 3]`` axis-angle, as SMPLest-X returns it.
    ``global_orient`` and ``transl`` are optional; leaving them out returns the
    body in its own frame, which is what a pose audit wants -- orientation and
    position are placement's problem, not the skeleton's.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    betas = np.asarray(betas, np.float32).reshape(len(betas), -1)
    body_pose = np.asarray(body_pose, np.float32).reshape(len(betas), -1)
    n = len(betas)

    out = np.empty((n, NUM_BODY_JOINTS, 3), np.float32)
    model = None
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        size = stop - start
        if model is None or size != batch:
            model = _load_model(model_dir, gender, size, device)
            batch = size
        kwargs = {
            "betas": torch.from_numpy(betas[start:stop, :10]).to(device),
            "body_pose": torch.from_numpy(body_pose[start:stop]).to(device),
        }
        if global_orient is not None:
            kwargs["global_orient"] = torch.from_numpy(
                np.asarray(global_orient, np.float32)[start:stop].reshape(size, 3)
            ).to(device)
        if transl is not None:
            kwargs["transl"] = torch.from_numpy(
                np.asarray(transl, np.float32)[start:stop].reshape(size, 3)
            ).to(device)
        with torch.no_grad():
            joints = model(**kwargs).joints
        out[start:stop] = joints[:, :NUM_BODY_JOINTS].cpu().numpy()
    _LOG.info("SMPL-X forward: %d frames -> %d body joints", n, NUM_BODY_JOINTS)
    return out


def looks_like_a_body(joints, *, parents=None) -> bool:
    """Cheap sanity check that a joint array really is the SMPL-X body tree.

    Worth calling before trusting any joint array whose provenance is a model
    output: it catches a mislabelled ordering immediately, where every
    downstream metric would otherwise report confident nonsense.
    """
    from nfl_gsplat.pose.forward_kinematics import SMPLX_BODY_PARENTS
    from nfl_gsplat.pose.plausibility import bone_lengths

    parents = SMPLX_BODY_PARENTS if parents is None else parents
    lengths = np.nanmedian(bone_lengths(joints, parents), axis=0)
    # No bone in a human body tree is under 2 cm or over 60 cm.
    return bool(np.all(lengths > 0.02) and np.all(lengths < 0.60))
