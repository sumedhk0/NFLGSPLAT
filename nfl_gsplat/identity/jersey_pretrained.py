"""A jersey-number reader on a PRETRAINED backbone, fine-tuned on this footage.

WHY. The from-scratch reader in jersey_cnn measured 25.1% per track against the
play's roster, on held-out plays, where generic easyocr measures 75%. Its own
post-mortem ruled out resolution and crop geometry and left two suspects: the
data volume (34k crops over ninety plays is thin for digits that must
generalise to unseen uniforms and stadiums) and the absence of any visual
prior. A network that already knows edges, strokes and textures from ImageNet
needs far fewer examples to learn what a 7 looks like on a moving shoulder.
This is the attempt that docstring asked for.

WHAT IS KEPT FROM jersey_cnn, deliberately: the two digit heads (tens with a
blank class, units), the [-1, 1] input convention, and the log-probability
interface that identity accumulates over a track and restricts to the roster.
Everything downstream is unchanged, so the comparison is like for like.

WHAT IS DIFFERENT: a ResNet-18 trunk with ImageNet weights, the 64 px crops
upsampled to 128 px (the trunk's strides eat a 64 px input to 2x2), ImageNet
normalisation inside the model so callers keep feeding [-1, 1], and a lower
learning rate on the trunk than on the fresh heads so the prior is refined
rather than overwritten.

MEASURE PER TRACK, AGAINST THE ROSTER. That is what identity uses, and it is
where the 25.1% and the 75% were measured. Per-crop numbers are printed too,
but a track pools evidence over dozens of crops and is the unit that matters.

MEASURED, 2026-09-02, 34k training crops, 640 held-out tracks, 6 epochs:

    pretrained resnet18   per crop 23.7%   per track 39.4%
    from-scratch cnn      per crop 18.5%   per track 25.2%   (same harness)
    easyocr                                per track 75%

The prior is worth fourteen points per track and reproduces the CNN's
recorded number to a tenth, so the harness is sound. It is not a replacement
for OCR. Training loss was still falling at the last epoch (2.58), so a
longer run will move it some; the gap to 75% is not a schedule. What this
reader may still be worth is as a SECOND signal alongside OCR -- they fail
differently -- which is the next thing to measure, in 07g, not here.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.identity.jersey_cnn import (
    normalise,
    predict_logprobs,
    split_number,
)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

INPUT_PX: int = 128          # what the trunk sees; crops are upsampled to this
TRUNK_LR: float = 1e-4
HEAD_LR: float = 1e-3


def build_model(dropout: float = 0.3, pretrained: bool = True):
    """ResNet-18 trunk, two digit heads. Torch imported lazily (CPU-safe module)."""
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as F

    class PretrainedJerseyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = (torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                       if pretrained else None)
            base = torchvision.models.resnet18(weights=weights)
            self.trunk = nn.Sequential(*list(base.children())[:-1])   # -> [N,512,1,1]
            self.drop = nn.Dropout(dropout)
            self.tens = nn.Linear(512, 11)      # 0-9 plus blank
            self.units = nn.Linear(512, 10)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            # Callers feed [-1, 1] (jersey_cnn.normalise); the trunk wants
            # ImageNet statistics at a size its strides can digest.
            x = (x + 1.0) / 2.0
            x = (x - self.mean) / self.std
            x = F.interpolate(x, size=(INPUT_PX, INPUT_PX), mode="bilinear",
                              align_corners=False)
            h = self.drop(self.trunk(x).flatten(1))
            return self.tens(h), self.units(h)

    torch.manual_seed(0)
    return PretrainedJerseyNet()


def train(crops, numbers, *, epochs: int = 6, batch: int = 128,
          device: str | None = None, val=None, pretrained: bool = True):
    """Fine-tune. ``val`` is an optional ``(crops, numbers)`` held-out set."""
    import torch
    from torch import nn

    from nfl_gsplat.identity.jersey_cnn import accuracy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=pretrained).to(device)
    heads = list(model.tens.parameters()) + list(model.units.parameters())
    opt = torch.optim.AdamW([
        {"params": model.trunk.parameters(), "lr": TRUNK_LR},
        {"params": heads, "lr": HEAD_LR},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[TRUNK_LR, HEAD_LR], total_steps=epochs * ((len(crops) + batch - 1) // batch))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    x = torch.from_numpy(normalise(crops))
    tens = torch.tensor([split_number(n)[0] for n in numbers], dtype=torch.long)
    units = torch.tensor([split_number(n)[1] for n in numbers], dtype=torch.long)
    n = len(x)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = x[idx].to(device, non_blocking=True)
            # The crops come from a synthesised body box whose framing wobbles;
            # digits never flip, so no mirroring. Shift, and jitter brightness
            # -- stadium lighting varies more than anything else across plays.
            if epoch:
                dx = int(torch.randint(-4, 5, (1,)).item())
                dy = int(torch.randint(-3, 4, (1,)).item())
                xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
                gain = 1.0 + 0.25 * (torch.rand(len(idx), 1, 1, 1, device=device) - 0.5)
                xb = (xb * gain).clamp_(-1.0, 1.0)
            pt, pu = model(xb)
            loss = (loss_fn(pt, tens[idx].to(device))
                    + loss_fn(pu, units[idx].to(device)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        msg = f"epoch {epoch + 1}/{epochs} loss {total / n:.4f}"
        if val is not None:
            msg += f"  val exact {accuracy(model, *val, device=device):.1%}"
        _LOG.info("jersey pretrained: %s", msg)
    return model


def play_of(track: str) -> str:
    """``'57583_000082_Endzone.mp4|H90'`` -> ``'57583_000082'``."""
    return "_".join(str(track).split("|")[0].split("_")[:2])


def track_accuracy(model, crops, numbers, tracks, *, device: str | None = None):
    """Per-track top-1 against that PLAY's roster, the way identity scores.

    Evidence is summed in log-probability over a track's crops; the answer is
    the roster number with the most. The roster of a play is every number
    that appears on a track of that play, in either view. Returns
    ``(per_track, per_crop, n_tracks)``.
    """
    numbers = np.asarray(numbers)
    tracks = np.asarray(tracks)
    lp = predict_logprobs(model, crops, device=device)      # [N, 11], [N, 10]
    lp_t, lp_u = lp
    plays = np.array([play_of(t) for t in tracks])
    roster = {p: sorted(set(numbers[plays == p].tolist())) for p in np.unique(plays)}

    def number_scores(lt, lu, ros):
        out = []
        for num in ros:
            t, u = split_number(int(num))
            out.append(lt[..., t] + lu[..., u])
        return np.asarray(out)

    crop_hits = 0
    for i in range(len(numbers)):
        ros = roster[plays[i]]
        s = number_scores(lp_t[i], lp_u[i], ros)
        crop_hits += int(ros[int(np.argmax(s))] == numbers[i])
    track_hits, n_tracks = 0, 0
    for tr in np.unique(tracks):
        idx = np.flatnonzero(tracks == tr)
        ros = roster[plays[idx[0]]]
        s = number_scores(lp_t[idx].sum(0), lp_u[idx].sum(0), ros)
        track_hits += int(ros[int(np.argmax(s))] == numbers[idx[0]])
        n_tracks += 1
    return track_hits / max(n_tracks, 1), crop_hits / max(len(numbers), 1), n_tracks
