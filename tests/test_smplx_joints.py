"""Guarding the joint-ordering bug that made every downstream length nonsense."""
import numpy as np
import pytest

from nfl_gsplat.pose.forward_kinematics import NUM_BODY_JOINTS, SMPLX_BODY_PARENTS
from nfl_gsplat.pose.smplx_joints import looks_like_a_body


def a_real_skeleton(T=10, seed=0):
    """A plausible body: every bone between 3 and 60 cm."""
    rng = np.random.default_rng(seed)
    parents = np.asarray(SMPLX_BODY_PARENTS)
    offsets = rng.uniform(0.08, 0.35, size=(NUM_BODY_JOINTS, 3))
    offsets *= rng.choice([-1.0, 1.0], size=offsets.shape)
    rest = np.zeros((NUM_BODY_JOINTS, 3))
    for j in range(1, NUM_BODY_JOINTS):
        rest[j] = rest[parents[j]] + offsets[j] / np.sqrt(3)
    return np.repeat(rest[None], T, axis=0)


def test_a_real_skeleton_is_accepted():
    assert looks_like_a_body(a_real_skeleton())


def test_the_mislabelled_ordering_is_rejected():
    """SMPLest-X's joints3d_cam[:22] is not the SMPL-X tree, and it shows.

    Measured on real output, reading those as the body gave a 1.256 m forearm
    and a 0.918 m spine link inside a skeleton spanning 1.03 m. Shuffling a real
    skeleton's joints reproduces exactly that signature.
    """
    seq = a_real_skeleton()
    rng = np.random.default_rng(1)
    shuffled = seq[:, rng.permutation(NUM_BODY_JOINTS)]
    assert not looks_like_a_body(shuffled)


def test_a_collapsed_skeleton_is_rejected():
    assert not looks_like_a_body(np.zeros((5, NUM_BODY_JOINTS, 3)))


def test_body_joints_needs_the_model_and_says_so():
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.pose.smplx_joints import body_joints

    pytest.importorskip("torch")
    pytest.importorskip("smplx")
    with pytest.raises(SetupError):
        body_joints(np.zeros((2, 10), np.float32),
                    np.zeros((2, 21, 3), np.float32),
                    model_dir="does/not/exist")
