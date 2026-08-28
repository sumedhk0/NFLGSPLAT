

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
