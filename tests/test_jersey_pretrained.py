"""The pretrained reader must plug into identity exactly where the CNN did."""
import numpy as np
import pytest

from nfl_gsplat.identity.jersey_pretrained import play_of

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def test_play_id_comes_off_the_track_name():
    assert play_of("57583_000082_Endzone.mp4|H90") == "57583_000082"
    assert play_of("57583_000082_Sideline.mp4|V7") == "57583_000082"


def test_it_takes_the_same_input_and_gives_the_same_heads_as_the_cnn():
    from nfl_gsplat.identity.jersey_cnn import normalise, predict_logprobs
    from nfl_gsplat.identity.jersey_pretrained import build_model

    model = build_model(pretrained=False)          # no download in tests
    crops = np.random.default_rng(0).integers(0, 255, (4, 64, 64, 3), np.uint8)
    x = torch.from_numpy(normalise(crops))
    assert x.min() >= -1.0 and x.max() <= 1.0
    pt, pu = model(x)
    assert pt.shape == (4, 11) and pu.shape == (4, 10)
    lt, lu = predict_logprobs(model, crops, device="cpu")
    assert lt.shape == (4, 11) and lu.shape == (4, 10)
    assert np.allclose(np.exp(lt).sum(1), 1.0, atol=1e-5)


def test_track_accuracy_pools_crops_and_restricts_to_the_roster(monkeypatch):
    """Two plays, two tracks each; the second track of play B is noisy per
    crop but right on the pooled evidence."""
    from nfl_gsplat.identity import jersey_pretrained as jp

    numbers = np.array([7, 7, 7, 22, 22, 22, 45, 45, 45, 88, 88, 88])
    tracks = np.array(["A_1_Sideline.mp4|H7"] * 3 + ["A_1_Endzone.mp4|H22"] * 3
                      + ["B_2_Sideline.mp4|H45"] * 3 + ["B_2_Endzone.mp4|H88"] * 3)
    crops = np.zeros((12, 64, 64, 3), np.uint8)

    # Per-crop log-probs: all right except one crop of the 88 track, which
    # says 45 -- pooled over the track, 88 still wins.
    lt = np.full((12, 11), -5.0)
    lu = np.full((12, 10), -5.0)
    said = numbers.copy()
    said[9] = 45
    for i, num in enumerate(said):
        t, u = jp.split_number(int(num))
        lt[i, t] = 0.0
        lu[i, u] = 0.0
    monkeypatch.setattr(jp, "predict_logprobs", lambda *_a, **_k: (lt, lu))

    per_track, per_crop, n_tracks = jp.track_accuracy(None, crops, numbers, tracks)
    assert n_tracks == 4
    assert per_track == 1.0
    assert per_crop == pytest.approx(11 / 12)
