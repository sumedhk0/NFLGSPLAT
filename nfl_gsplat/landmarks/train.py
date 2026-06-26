"""Heatmap keypoint training. GPU jobs run on PACE `embers` (preemptible) → every
epoch is checkpointed and resumable."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _masked_mse(pred, target, vis):
    w = vis[:, :, None, None]
    return (w * (pred - target) ** 2).sum() / (w.sum().clamp_min(1.0) * pred.shape[-1] * pred.shape[-2])


def train(label_json, frames_dir, schema, *, out_dir, epochs, batch_size, lr,
          device="cuda", resume=True) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from nfl_gsplat.landmarks.dataset import LandmarkDataset
    from nfl_gsplat.landmarks.model import LandmarkNet

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = LandmarkDataset(label_json, frames_dir, schema, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    net = LandmarkNet(schema.num_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start_ep, best = 0, float("inf")
    last = out / "ckpt_last.pt"
    if resume and last.exists():
        st = torch.load(last, map_location=device)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        start_ep, best = st["epoch"] + 1, st["best"]
    for ep in range(start_ep, epochs):
        net.train(); total = 0.0
        for img, heat, vis in dl:
            img, heat, vis = img.to(device), heat.to(device), vis.to(device)
            pred = net(img)
            loss = _masked_mse(pred, heat, vis)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": min(best, total),
                    "classes": schema.class_names()}, last)
        if total < best:
            best = total
            torch.save({"net": net.state_dict(), "classes": schema.class_names()},
                       out / "ckpt_best.pt")
    return out / "ckpt_best.pt"


def _collate(batch):
    import torch
    imgs = torch.from_numpy(np.stack([b[0] for b in batch]))
    heats = torch.from_numpy(np.stack([b[1] for b in batch]))
    vis = torch.from_numpy(np.stack([b[2] for b in batch]))
    return imgs, heats, vis


def main():
    import argparse
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yard-min", type=float, default=-25.0)
    ap.add_argument("--yard-max", type=float, default=25.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    schema = LandmarkSchema(yard_min=a.yard_min, yard_max=a.yard_max)
    ck = train(a.label, a.frames, schema, out_dir=a.out, epochs=a.epochs,
               batch_size=a.batch_size, lr=a.lr, device=a.device, resume=a.resume)
    print(f"best checkpoint: {ck}")


if __name__ == "__main__":
    main()
