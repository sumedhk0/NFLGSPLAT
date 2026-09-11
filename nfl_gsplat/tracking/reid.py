"""Appearance embeddings per detection, for re-identification while linking.

WHY. Linking players on the turf by position alone is level with the 2D
tracker at this placement noise (scripts/07j): 3-7 pieces per player, and a
tenth of tracks under 60% one player. Every cheap cue has been measured and
rejected -- per-detection team colour twice, looser gates -- because a cue
that is wrong one frame in five breaks a track that frame. What the 2D
tracker has and the turf does not is APPEARANCE, and what a wrong colour
label lacks is a soft way to be wrong: an embedding compared by cosine
similarity moves a cost, it does not veto an assignment.

WHAT. A pretrained ImageNet ResNet-18 trunk (the same prior that lifted the
jersey reader from 25% to 42%) over each detection's torso crop, 512-d,
L2-normalised. No training: two crops of the same player a few frames apart
are near in this space and two players in the same uniform are less near,
which is all a tie-breaker needs. Batched on the GPU, one pass over the
video per view.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

CROP_PX: int = 64
# Torso band of the person box: below the helmet, above the turf.
TORSO_TOP: float = 0.2
TORSO_BOTTOM: float = 0.65


def torso_crop(img, box, *, size: int = CROP_PX):
    """``[size, size, 3]`` uint8 RGB of the box's torso band, or None."""
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bh = y2 - y1
    ya, yb = int(max(0, y1 + TORSO_TOP * bh)), int(min(h, y1 + TORSO_BOTTOM * bh))
    xa, xb = int(max(0, x1)), int(min(w, x2))
    if yb - ya < 4 or xb - xa < 4:
        return None
    crop = img[ya:yb, xa:xb, ::-1]                       # BGR -> RGB
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _trunk(device, weights=None):
    """The embedding network: ImageNet trunk, or the football-trained one
    from scripts/07k_train_reid.py (trunk + 128-d head) when ``weights`` is
    given. Returns ``(net, mean, std)`` with ``net(x) -> [N, D]``."""
    import torch
    import torchvision
    from torch import nn

    base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    trunk = nn.Sequential(*list(base.children())[:-1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if weights is None:
        net = nn.Sequential(trunk, nn.Flatten(1))
    else:
        ckpt = torch.load(str(weights), map_location=device)
        trunk.load_state_dict(ckpt["trunk"])
        head = nn.Sequential(nn.Linear(512, int(ckpt["embed_dim"])),
                             nn.BatchNorm1d(int(ckpt["embed_dim"])))
        head.load_state_dict(ckpt["embed"])
        net = nn.Sequential(trunk, nn.Flatten(1), head)
        _LOG.info("re-id: football-trained weights from %s", weights)
    return net.to(device).eval(), mean, std


def embed_crops(crops, *, device: str | None = None, batch: int = 256,
                weights=None) -> np.ndarray:
    """``[N, D]`` unit-norm embeddings of ``[N, H, W, 3]`` uint8 RGB crops."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trunk, mean, std = _trunk(device, weights)
    crops = np.asarray(crops, np.uint8)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            if weights is not None:
                # The trained head was fitted on 64 px crops upsampled to 128.
                x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear",
                                                    align_corners=False)
            f = trunk(x)
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def embed_detections(video_path, df_view, *, device: str | None = None, weights=None):
    """``[len(df_view), D]`` embeddings (NaN rows where no crop), one video pass.

    ``df_view`` needs ``frame`` and ``bbox_x1..bbox_y2`` columns.
    """
    import cv2

    frames = df_view["frame"].to_numpy()
    boxes = df_view[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    crops, rows = [], []
    cap = cv2.VideoCapture(str(video_path))
    for f in np.unique(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        for r in np.flatnonzero(frames == f):
            c = torso_crop(img, boxes[r])
            if c is not None:
                crops.append(c)
                rows.append(r)
    cap.release()
    emb = embed_crops(np.stack(crops), device=device, weights=weights) if crops else None
    dim = emb.shape[1] if emb is not None else 512
    feats = np.full((len(df_view), dim), np.nan, np.float32)
    if emb is not None:
        feats[np.asarray(rows)] = emb
    _LOG.info("re-id: embedded %d of %d detections", len(rows), len(df_view))
    return feats
