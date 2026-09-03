"""The football-trained checkpoint (scripts/07k) loads into tracking.reid."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from nfl_gsplat.tracking import reid  # noqa: E402


def _checkpoint(path, dim=16):
    from torch import nn

    trunk, _mean, _std = reid._trunk("cpu")
    head = nn.Sequential(nn.Linear(512, dim), nn.BatchNorm1d(dim))
    torch.save({"trunk": trunk[0].state_dict(), "embed": head.state_dict(),
                "embed_dim": dim}, path)


def test_trained_weights_give_head_dim_unit_vectors(tmp_path):
    ck = tmp_path / "reid.pt"
    _checkpoint(ck, dim=16)
    crops = np.random.default_rng(0).integers(0, 255, size=(5, reid.CROP_PX, reid.CROP_PX, 3),
                                              dtype=np.uint8)
    e = reid.embed_crops(crops, device="cpu", weights=ck)
    assert e.shape == (5, 16)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    e0 = reid.embed_crops(crops, device="cpu")
    assert e0.shape == (5, 512)                          # ImageNet path unchanged
