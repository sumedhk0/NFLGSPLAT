#!/usr/bin/env python
"""Train a re-identification embedding on football crops, and measure it.

Generic ImageNet embeddings of ~50 px torso crops did nothing for linking
(scripts/07j): two team-mates in one uniform look the same to them. The
helmet set has what a trained embedding needs -- tens of thousands of crops
labelled by player, across both cameras, held out by PLAY -- and
07h_build_jersey_dataset.py already cut them (jersey_crops_det2.npz: crops,
splits, tracks as ``video|label``).

WHAT IS TRAINED. The same ImageNet ResNet-18 trunk, a 128-d embedding head,
and a softmax classifier over PLAYER keys (play + jersey label, so the two
views of one player share a class) -- but the softmax runs over the players
OF THE SAME PLAY only, and every batch is one play. Measured first with the
plain classifier over all 997 players: held-out same-camera retrieval fell
from 39.5% (ImageNet, untrained) to 34.2%, because classes that each live in
one play are told apart by the play (uniform, lighting, turf), which is
trivial and useless. The linker's question is always within one play, so the
loss is asked that question. The embedding is the unit-normalised head
output; the classifier is thrown away.

WHAT IS MEASURED, on held-out plays, the way the linker will use it: within
one play and one camera, does a crop's nearest neighbour (cosine, every other
crop of that video, itself excluded) belong to the same player? The crops
are 40 frames spread over the play, so the neighbour is the same player a
fraction of a second later, or a team-mate in the same uniform -- which is
the linker's question. Chance is about 1/22. Reported for the ImageNet trunk
untrained and for the trained one, on the same crops; if trained is not far
above untrained, do not put it in the linker.

Cross-CAMERA retrieval (nearest crop in the other view of the play) is
printed too, for the record: the first version of this script measured only
that, by accident, and read "no gain" at 2%.

RESULT (2026-09-02, 34k crops / 997 players train, 12k / 330 / 15 plays held
out): every recipe lands BELOW the untouched ImageNet trunk at the linker's
question. Same-camera nearest neighbour same player: ImageNet 39.5%; softmax
over all players 34.2%; softmax within the play 33.7%; head only on a frozen
trunk 32.1% (so it is not fine-tuning forgetting -- identity supervision at
0.25 s spacing pulls the space away from the instant-appearance matches that
consecutive-frame linking lives on). Cross-camera 11.0% -> 12-15%. The
ImageNet trunk was already measured neutral in the linker (07j), so the
trained weights were not run there. Kept so the next idea starts from the
numbers, not from scratch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
EMBED_D = 128


def player_key(track: str) -> str:
    """``'57583_000082_Endzone.mp4|H90'`` -> ``'57583_000082|H90'``."""
    video, label = str(track).split("|")
    return "_".join(video.split("_")[:2]) + "|" + label


def play_of(track: str) -> str:
    return "_".join(str(track).split("|")[0].split("_")[:2])


def build_model(n_classes: int, device):
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as F

    class ReID(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.trunk = nn.Sequential(*list(base.children())[:-1])
            self.embed = nn.Sequential(nn.Linear(512, EMBED_D), nn.BatchNorm1d(EMBED_D))
            self.cls = nn.Linear(EMBED_D, n_classes)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def features(self, x):                       # x uint8-derived float [0,1], [N,3,64,64]
            x = (x - self.mean) / self.std
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
            return F.normalize(self.embed(self.trunk(x).flatten(1)), dim=1)

        def forward(self, x):
            e = self.features(x)
            return self.cls(e), e

    torch.manual_seed(0)
    return ReID().to(device)


def embed_all(model, crops, device, *, batch=512, untrained_trunk=None):
    """Unit-norm embeddings; with ``untrained_trunk`` use ImageNet features (512-d)."""
    import torch
    from torch.nn import functional as F

    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
            if untrained_trunk is not None:
                x = (x - model.mean) / model.std
                x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
                e = F.normalize(untrained_trunk(x).flatten(1), dim=1)
            else:
                e = model.features(x)
            out.append(e.cpu().numpy())
    return np.concatenate(out)


def retrieval(emb, keys, groups, *, cross=None, max_per_group=1500, seed=0):
    """Top-1 same-player rate of each crop's nearest neighbour within its
    ``group`` (a video), itself excluded. With ``cross`` (a play id per crop)
    the group is the play and only crops from a DIFFERENT video are eligible.
    Subsampled per group for speed."""
    rng = np.random.default_rng(seed)
    hits, n = 0, 0
    scope = cross if cross is not None else groups
    for g in np.unique(scope):
        idx = np.flatnonzero(scope == g)
        if len(idx) > max_per_group:
            idx = rng.choice(idx, max_per_group, replace=False)
        e, k, v = emb[idx], keys[idx], groups[idx]
        sim = e @ e.T
        np.fill_diagonal(sim, -np.inf)
        if cross is not None:
            sim[v[:, None] == v[None, :]] = -np.inf            # other camera only
        nn_ = sim.argmax(1)
        ok = np.isfinite(sim[np.arange(len(idx)), nn_])
        hits += int((k[nn_][ok] == k[ok]).sum())
        n += int(ok.sum())
    return hits / max(n, 1), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=Path("data/helmet/jersey_crops_det2.npz"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("data/weights/reid_resnet18.pt"))
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="train the head only (does fine-tuning forget what ImageNet knew?)")
    args = ap.parse_args()

    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.crops, allow_pickle=True)
    crops, splits, tracks = z["crops"], z["splits"], z["tracks"]
    keys = np.array([player_key(t) for t in tracks])
    plays = np.array([play_of(t) for t in tracks])
    videos = np.array([str(t).split("|")[0] for t in tracks])
    tr, te = splits == 0, splits == 1
    classes = {k: i for i, k in enumerate(sorted(set(keys[tr])))}
    y = np.array([classes.get(k, -1) for k in keys])
    print(f"{tr.sum()} train crops over {len(classes)} players; {te.sum()} held-out crops "
          f"over {len(set(keys[te]))} players in {len(set(plays[te]))} plays")

    model = build_model(len(classes), device)
    # Untrained baseline: the ImageNet trunk's own 512-d features.
    base_trunk = build_model(len(classes), device).trunk.eval()
    e0 = embed_all(model, crops[te], device, untrained_trunk=base_trunk)
    r0, n0 = retrieval(e0, keys[te], videos[te])
    x0, _ = retrieval(e0, keys[te], videos[te], cross=plays[te])
    print(f"untrained ImageNet trunk: same-camera NN same player {r0:.1%} ({n0} crops); "
          f"cross-camera {x0:.1%}")

    groups = [{"params": list(model.embed.parameters()) + list(model.cls.parameters()), "lr": 1e-3}]
    if args.freeze_trunk:
        for q in model.trunk.parameters():
            q.requires_grad_(False)
    else:
        groups.append({"params": model.trunk.parameters(), "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    # No label smoothing: it would put target mass on the masked (-inf) classes.
    loss_fn = nn.CrossEntropyLoss()
    x_tr = crops[tr]
    y_tr = torch.tensor(y[tr], dtype=torch.long)
    play_tr = plays[tr]
    # One batch = one play: the classes on offer are that play's players.
    by_play = {pl: np.flatnonzero(play_tr == pl) for pl in np.unique(play_tr)}
    mask_of = {pl: torch.tensor(np.isin(np.arange(len(classes)), np.unique(y[tr][ix])),
                                device=device) for pl, ix in by_play.items()}
    rng = np.random.default_rng(0)
    n = len(x_tr)
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_trunk:
            model.trunk.eval()                      # keep BatchNorm statistics
        total, seen = 0.0, 0
        order = list(by_play)
        rng.shuffle(order)
        for pl in order:
            ix = by_play[pl]
            rng.shuffle(ix)
            for i in range(0, len(ix), args.batch):
                idx = ix[i:i + args.batch]
                if len(idx) < 8:
                    continue
                xb = torch.from_numpy(x_tr[idx]).to(device).permute(0, 3, 1, 2).float() / 255.0
                if epoch:
                    dx = int(torch.randint(-4, 5, (1,)).item())
                    xb = torch.roll(xb, shifts=dx, dims=3)
                    gain = 1.0 + 0.2 * (torch.rand(len(idx), 1, 1, 1, device=device) - 0.5)
                    xb = (xb * gain).clamp_(0, 1)
                logits, _ = model(xb)
                logits = logits.masked_fill(~mask_of[pl][None, :], float("-inf"))
                loss = loss_fn(logits, y_tr[idx].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
                seen += len(idx)
        n = max(seen, 1)
        model.eval()
        e1 = embed_all(model, crops[te], device)
        r1, _ = retrieval(e1, keys[te], videos[te])
        x1, _ = retrieval(e1, keys[te], videos[te], cross=plays[te])
        _LOG.info("re-id: epoch %d/%d loss %.3f  held-out same-camera NN same player %.1f%% "
                  "(cross-camera %.1f%%)", epoch + 1, args.epochs, total / n, 100 * r1, 100 * x1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trunk": model.trunk.state_dict(), "embed": model.embed.state_dict(),
                "embed_dim": EMBED_D}, args.out)
    print(f"\nheld-out within-play nearest-neighbour same player: untrained {r0:.1%} -> "
          f"trained {r1:.1%}   (chance ~{100 / max(1, len(set(keys[te])) / len(set(plays[te]))):.1f}%)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
