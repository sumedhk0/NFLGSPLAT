"""Draw cached joints2d onto the frame they came from, and LOOK at the result.

Written because numbers lied. The triangulation this feeds reported 4.5 px
reprojection with 95% of joints valid while the reconstruction was upside down;
every internal consistency measure agreed with a wrong answer. One picture shows
it in a glance: a correct run puts a recognisable human skeleton inside the green
detection box, and 94% of joints land inside it.

Usage:
    python scripts/diag_joints2d.py <cam> <pose_cache.pkl> <out.png> [play_dir]

Run this after ANY change to cropping, to the bbox-derived camera, or to the
joint indexing -- those are exactly the changes whose failures look healthy in
aggregate statistics.
"""
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:/Users/sumedh/NFLGSPLAT")
from nfl_gsplat.utils.video import iter_frames

PD_ = r"data/2025/week_04/SEA_at_AZ/play_001"
CAM, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
if len(sys.argv) > 4:
    PD_ = sys.argv[4]
cache = pickle.load(open(CACHE, "rb"))["frames"]
frame = sorted(cache)[len(cache) // 3]

# SMPL-X body skeleton, for limbs that read as a person rather than a dot cloud.
# SMPL-X body kinematic tree, joint -> parent, for joints 0..21. Getting this
# wrong draws a tangle over correct joints, which is worse than useless: it
# looks like broken data and hides the real thing this tool exists to show.
# The collars hang off spine3 (9), NOT the neck (12).
PARENT = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
          11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16,
          19: 17, 20: 18, 21: 19}
BONES = [(child, parent) for child, parent in PARENT.items()]

img = None
for idx, rgb in iter_frames(PD_ + f"/{CAM}.mp4", stride=1):
    if idx == frame:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        break

tiles = []
for tid, rec in list(cache[frame].items())[:6]:
    x1, y1, x2, y2 = [int(v) for v in rec["bbox"]]
    j2 = np.asarray(rec["joints2d"])[:22]
    pad = 30
    sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
    sx2, sy2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    tile = img[sy1:sy2, sx1:sx2].copy()
    cv2.rectangle(tile, (x1 - sx1, y1 - sy1), (x2 - sx1, y2 - sy1),
                  (60, 220, 60), 2)
    for a, b in BONES:
        pa = (int(j2[a, 0] - sx1), int(j2[a, 1] - sy1))
        pb = (int(j2[b, 0] - sx1), int(j2[b, 1] - sy1))
        cv2.line(tile, pa, pb, (40, 40, 235), 2)
    for k, (u, v) in enumerate(j2):
        c = (255, 240, 60) if k == 15 else (235, 235, 235)   # head in cyan
        cv2.circle(tile, (int(u - sx1), int(v - sy1)), 3, c, -1)
    h = 320
    scale = h / max(1, tile.shape[0])
    tiles.append(cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), h)))

pad_w = max(t.shape[1] for t in tiles)
canvas = np.full((320, pad_w * len(tiles), 3), 24, np.uint8)
for i, t in enumerate(tiles):
    canvas[:, i * pad_w:i * pad_w + t.shape[1]] = t
cv2.imwrite(OUT, canvas)
print(f"{CAM} frame {frame}: wrote {OUT} ({len(tiles)} players; "
      "green = detection box, cyan dot = head joint)")
