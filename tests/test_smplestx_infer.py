

# --- the strict=False residue must be body-model constants, nothing else ------
# SMPLest-X loads with strict=False and prints "Please check manually" every
# run. That warning cannot tell a fully loaded network from one missing half
# its encoder, and a half-loaded network still returns plausible poses -- the
# exact failure that goes unnoticed. Measured on smplest_x_h: 519/519 tensors
# match, the only residue being the 17 SMPL-X body-model constants.

class _FakeModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def _write_ckpt(tmp_path, state):
    import torch
    p = tmp_path / "ckpt.pth.tar"
    torch.save({"epoch": 3, "network": state}, p)
    return p


def test_body_model_constants_are_not_reported_missing(tmp_path):
    """The SMPL-X layer's buffers come from the .npz, never from a checkpoint."""
    import torch

    from nfl_gsplat.pose.smplestx_infer import checkpoint_coverage
    model = _FakeModel({
        "module.encoder.weight": torch.zeros(4, 4),
        "module.smplx_layer.v_template": torch.zeros(10475, 3),
        "module.smplx_layer.lbs_weights": torch.zeros(10475, 55),
    })
    ckpt = _write_ckpt(tmp_path, {"module.encoder.weight": torch.zeros(4, 4)})
    missing, unexpected, mismatched = checkpoint_coverage(model, ckpt)
    assert (missing, unexpected, mismatched) == ([], [], [])


def test_a_missing_learned_parameter_fails_loud(tmp_path):
    """The failure this exists to catch: a real weight silently absent."""
    import pytest
    import torch

    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.pose.smplestx_infer import verify_checkpoint_coverage
    model = _FakeModel({
        "module.encoder.blocks.0.attn.qkv.weight": torch.zeros(8, 4),
        "module.decoder.weight": torch.zeros(4, 4),
    })
    ckpt = _write_ckpt(tmp_path, {"module.decoder.weight": torch.zeros(4, 4)})
    with pytest.raises(SetupError, match="absent from the checkpoint"):
        verify_checkpoint_coverage(model, ckpt)


def test_a_shape_mismatch_fails_loud(tmp_path):
    import pytest
    import torch

    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.pose.smplestx_infer import verify_checkpoint_coverage
    model = _FakeModel({"module.decoder.weight": torch.zeros(4, 4)})
    ckpt = _write_ckpt(tmp_path, {"module.decoder.weight": torch.zeros(8, 4)})
    with pytest.raises(SetupError, match="shape mismatch"):
        verify_checkpoint_coverage(model, ckpt)


def test_data_parallel_prefix_is_stripped_from_both_sides(tmp_path):
    """Stripping only one side reported all 536 tensors missing -- a naming
    artefact that looks like a catastrophically broken load."""
    import torch

    from nfl_gsplat.pose.smplestx_infer import (checkpoint_coverage,
                                                strip_data_parallel)
    assert strip_data_parallel("module.encoder.w") == "encoder.w"
    assert strip_data_parallel("encoder.w") == "encoder.w"
    # model wrapped, checkpoint not -- must still match
    model = _FakeModel({"module.encoder.w": torch.zeros(2, 2)})
    ckpt = _write_ckpt(tmp_path, {"encoder.w": torch.zeros(2, 2)})
    assert checkpoint_coverage(model, ckpt) == ([], [], [])


def test_a_clean_checkpoint_passes(tmp_path):
    import torch

    from nfl_gsplat.pose.smplestx_infer import verify_checkpoint_coverage
    model = _FakeModel({"module.encoder.w": torch.zeros(2, 2)})
    ckpt = _write_ckpt(tmp_path, {"module.encoder.w": torch.zeros(2, 2)})
    verify_checkpoint_coverage(model, ckpt)      # must not raise


# --- joints2d must land on the player whatever size the crop is ---------------
# The projection produces CROP-pixel coordinates and then adds the box's corner
# in the full frame. That is only correct when the crop IS the raw box pixels.
# 05c hands in 192x256 crops, so every skeleton landed up to 192 px right and
# 256 px below its player: 10% of joints inside their own box against the 94%
# this pipeline measures when correct, and an upside-down 3D reconstruction that
# reprojection error and joint validity both called healthy.

def test_joints2d_are_scaled_from_crop_pixels_to_the_original_box():
    """A resized crop must not shift the joints off the player."""
    import numpy as np
    import pytest

    # A joint at the centre of the crop belongs at the centre of the box.
    crop_h, crop_w = 256, 192
    x1, y1, x2, y2 = 100.0, 200.0, 140.0, 360.0        # 40 x 160 box
    crop_xy = np.array([[crop_w / 2, crop_h / 2]])

    sx = (x2 - x1) / crop_w
    sy = (y2 - y1) / crop_h
    got = crop_xy * np.array([sx, sy]) + np.array([x1, y1])

    assert got[0][0] == pytest.approx(0.5 * (x1 + x2))
    assert got[0][1] == pytest.approx(0.5 * (y1 + y2))
    # The un-scaled version -- the bug -- lands clear of the box entirely,
    # further from the true point than the player is wide.
    bad = crop_xy + np.array([x1, y1])
    assert bad[0][0] > x2
    miss = float(np.linalg.norm(bad[0] - got[0]))
    assert miss > (x2 - x1)


def test_an_unresized_crop_is_unchanged_by_the_scaling():
    """The fix must not disturb callers who already pass raw box pixels."""
    import numpy as np
    import pytest

    x1, y1, x2, y2 = 10.0, 20.0, 60.0, 220.0
    crop_h, crop_w = int(y2 - y1), int(x2 - x1)
    crop_xy = np.array([[7.0, 30.0]])
    sx = (x2 - x1) / crop_w
    sy = (y2 - y1) / crop_h
    got = crop_xy * np.array([sx, sy]) + np.array([x1, y1])
    assert got[0] == pytest.approx([x1 + 7.0, y1 + 30.0])
