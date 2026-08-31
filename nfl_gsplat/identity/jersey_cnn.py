"""A jersey-number reader trained on this footage, instead of generic OCR.

Generic OCR was never meant for this: small, motion-blurred, curved digits on a
moving shoulder at sixty metres. Measured on the Helmet Assignment set, easyocr
gave some read for 87% of tracks but the right number on top for only 75% of
those. The training data to do better already exists -- 952k labelled helmet
boxes, each carrying the player's number.

TWO DIGIT HEADS, NOT 100 CLASSES. A jersey is 0-99, so a flat 100-way softmax
looks natural, but it has to learn "18" and "19" as unrelated symbols and gets
no help from every other number sharing a 1. Predicting the tens and units
separately shares all the digit evidence, and a blank tens class covers the
single-digit numbers. It also degrades the way the problem does: half an
occluded number still gives a confident units digit.

THE OUTPUT IS A DISTRIBUTION, and that matters more than the argmax. Identity
does not need this model to name a player -- it needs evidence to accumulate
over a track and be weighed against the 22 numbers actually on the field.
``number_logprobs`` returns exactly that, restricted to a roster, which drops
into the same assignment the OCR path uses.

Deliberately small. The crops are 64x64 and the signal is a couple of large
digits; a bigger network would fit the turf and the jersey colour instead, and
this has to run over thousands of crops per play.

MEASURED, AND IT IS NOT GOOD ENOUGH YET. Trained on 48k crops and scored on
17k from HELD-OUT PLAYS: 10.1% exact number, 22.5% top-1 against a 22-number
roster (chance 4.5%). Real, but far from usable.

The interesting part is WHY, because it is not resolution. Accuracy by helmet
size runs 6.0% / 7.3% / 12.3% / 7.6% across 8-14, 14-20, 20-30 and 30+ px
helmets -- it PEAKS in the middle and falls for the biggest, closest players.
Small crops being hard is expected; large ones being hard is not, and it points
at the crop itself rather than the pixels in it. The body box is synthesised
from the helmet at a fixed 6.5-heads ratio, and that ratio cannot hold across
scale and posture: for a near player the box runs off the frame or past the
number entirely. Fixing the crop -- a real person detector, or a ratio that
adapts to posture -- is the next thing to try, not a bigger network.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

CROP_PX: int = 64
BLANK: int = 10          # the tens digit of a single-digit number


def split_number(number: int) -> tuple[int, int]:
    """``(tens, units)`` with BLANK for a number below ten."""
    number = int(number)
    if number < 0 or number > 99:
        raise ValueError(f"jersey number out of range: {number}")
    if number < 10:
        return BLANK, number
    return number // 10, number % 10


def join_number(tens: int, units: int) -> int:
    return int(units) if int(tens) == BLANK else int(tens) * 10 + int(units)


def build_model(dropout: float = 0.3):
    """A small two-head CNN. Torch is imported lazily so this module stays CPU-safe."""
    import torch
    from torch import nn

    class JerseyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            def block(i, o):
                return nn.Sequential(
                    nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2))
            self.trunk = nn.Sequential(
                block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout))
            self.tens = nn.Linear(128, 11)      # 0-9 plus blank
            self.units = nn.Linear(128, 10)

        def forward(self, x):
            h = self.trunk(x)
            return self.tens(h), self.units(h)

    torch.manual_seed(0)
    return JerseyNet()


def normalise(crops):
    """uint8 ``[N, H, W, 3]`` -> float32 ``[N, 3, H, W]`` in [-1, 1]."""
    arr = np.asarray(crops, np.float32) / 127.5 - 1.0
    if arr.ndim == 3:
        arr = arr[None]
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


def train(crops, numbers, *, epochs: int = 8, batch: int = 256,
          lr: float = 3e-4, device: str | None = None, val=None):
    """Fit the reader. ``val`` is an optional ``(crops, numbers)`` held-out set."""
    import torch
    from torch import nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

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
            # Light augmentation: these crops are cut from a synthesised body
            # box, so their framing wobbles in exactly this way at inference.
            if epoch:
                shift = int(torch.randint(-3, 4, (1,)).item())
                xb = torch.roll(xb, shifts=shift, dims=3)
            pt, pu = model(xb)
            loss = (loss_fn(pt, tens[idx].to(device))
                    + loss_fn(pu, units[idx].to(device)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        msg = f"epoch {epoch + 1}/{epochs} loss {total / n:.4f}"
        if val is not None:
            msg += f"  val exact {accuracy(model, *val, device=device):.1%}"
        _LOG.info("jersey cnn: %s", msg)
    return model


def predict_logprobs(model, crops, *, device: str | None = None,
                     batch: int = 512):
    """``(log P(tens) [N,11], log P(units) [N,10])``."""
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(normalise(crops))
    outs_t, outs_u = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            pt, pu = model(x[i:i + batch].to(device))
            outs_t.append(torch.log_softmax(pt, dim=1).cpu().numpy())
            outs_u.append(torch.log_softmax(pu, dim=1).cpu().numpy())
    return np.concatenate(outs_t), np.concatenate(outs_u)


def number_logprobs(model, crops, roster, *, device: str | None = None):
    """``[N, len(roster)]`` log-likelihood of each ROSTER number per crop.

    Restricted to the roster on purpose. The model does not have to decide what
    the number is in the abstract -- only which of the 22 out there it looks
    most like, which is a far easier question and the one identity actually
    asks.
    """
    lt, lu = predict_logprobs(model, crops, device=device)
    roster = [int(j) for j in roster]
    out = np.empty((len(lt), len(roster)), np.float32)
    for j, number in enumerate(roster):
        t, u = split_number(number)
        out[:, j] = lt[:, t] + lu[:, u]
    return out


def accuracy(model, crops, numbers, *, device: str | None = None) -> float:
    """Fraction of crops whose exact number is the argmax of both heads."""
    lt, lu = predict_logprobs(model, crops, device=device)
    got = [join_number(t, u) for t, u in zip(lt.argmax(1), lu.argmax(1))]
    return float(np.mean(np.asarray(got) == np.asarray(numbers, int)))


def save(model, path):
    import torch

    torch.save(model.state_dict(), str(path))


def load(path, *, device: str | None = None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(str(path), map_location=device))
    return model.to(device).eval()
