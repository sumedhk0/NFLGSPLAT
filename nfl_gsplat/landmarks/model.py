"""Compact UNet keypoint heatmap network (one sigmoid channel per landmark)."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class LandmarkNet(nn.Module):
    def __init__(self, num_classes: int, *, stride: int = 4):
        super().__init__()
        assert stride == 4, "decoder returns 1/4 resolution"
        self.e1 = _block(3, 32)
        self.e2 = _block(32, 64)
        self.e3 = _block(64, 128)
        self.e4 = _block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = _block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = _block(128, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        up3 = nn.functional.interpolate(self.up3(e4), size=e3.shape[2:],
                                        mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up3, e3], 1))
        up2 = nn.functional.interpolate(self.up2(d3), size=e2.shape[2:],
                                        mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([up2, e2], 1))
        out = self.head(d2)
        out = nn.functional.interpolate(
            out, size=(x.shape[2] // 4, x.shape[3] // 4), mode="bilinear", align_corners=False)
        return torch.sigmoid(out)
