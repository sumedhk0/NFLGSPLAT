#!/usr/bin/env python
"""Fit each body's Gaussian appearance to the footage (M8, step 3).

Input is the world-mode pose cache from 05f (SMPL-X params + transl per
frame per player) and the play-dir's cameras and videos. For each player
with enough posed frames: the body is placed per frame exactly as 05d does
(SMPL-X forward + transl), every posed frame in BOTH views becomes an
observation, every HOLDOUT-th frame is held out, the appearance is
initialised from the median footage texture (what 05d renders today) and
fitted with compositing.fit_appearance.

What is printed is the acceptance measure from the design doc: held-out
crop L1 for the median texture against the fitted appearance, per body and
over bodies (median, worst). Colour of vertices no camera ever faced is
reported so turf bleed shows up. Nothing replaces the median texture in the
render unless 05d is told to (--fitted-appearance <dir>).

Defaults are the winner of a four-variant ablation on one body (play 2,
player 10, 76 train / 20 held-out crops, held-out L1 with a test-time shift
for both sides): 50 it lr 0.02 tv 0.01 +3%; the same without the training
shift +0%; 150 it tv 0.05 +4%; 150 it tv 0.05 lr 0.005 +6%. The nuisance is
what makes the fit beat the median texture; the gain is modest because the
median texture is a strong baseline at 140-px bodies.

Runs in the smplx env (SMPL-X forward; torch with CUDA).

    python scripts/05i_fit_appearance.py --play-dir <P> --poses <P>/poses_refit.json \\
        --out-dir <P>/appearance --iters 200
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
MIN_FRAMES = 8
BODY_RGB = (0.7, 0.7, 0.7)
# A sample this close (RGB, 0..1) to the frame's turf colour is turf, not the
# body: at 140 px a limb is 3-5 px wide and most of its vertices sample
# pixels mixed with grass, which made whole bodies khaki. Dropped before the
# median; a vertex left with nothing takes the body's dominant non-turf colour.
TURF_DIST = 0.12


class FrameCache:
    """Frames of both views, read once each, RGB float32 in [0, 1]."""

    def __init__(self, play_dir: Path, cams):
        import cv2

        self.caps = {c: cv2.VideoCapture(str(play_dir / f"{c}.mp4")) for c in cams}
        self.frames: dict[tuple[str, int], np.ndarray] = {}

    def get(self, cam: str, f: int):
        import cv2

        key = (cam, int(f))
        if key not in self.frames:
            cap = self.caps[cam]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                raise SetupError(f"{cam}.mp4 has no frame {f}")
            self.frames[key] = img[..., ::-1].astype(np.float32) / 255.0
        return self.frames[key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--poses", required=True, type=Path, help="05f world-mode cache")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--tv-weight", type=float, default=0.05)
    ap.add_argument("--holdout", type=int, default=5, help="every n-th frame is held out")
    ap.add_argument("--min-facing", type=float, default=0.5,
                    help="cosine a vertex must face the camera by to be sampled for the START "
                         "colour; 0.5 cut the start greenness by a third against 0.1 (turf at "
                         "the silhouette) and raised the held-out gain")
    ap.add_argument("--limit", type=int, default=0, help="fit at most this many bodies")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-translation", dest="translation", action="store_false",
                    help="no per-frame shift nuisance during the fit")
    ap.add_argument("--from-timeline", action="store_true",
                    help="fit EVERY timeline player (render.play_timeline: fused, single-view "
                         "and default-posed) from its placed vertices, not only the fused "
                         "refit cache; the render (05k) places bodies the same way")
    ap.add_argument("--no-turf-mask", dest="turf_mask", action="store_false",
                    help="keep samples near the frame's turf colour in the start colour")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip players that already have appearance_<pid>.npz in --out-dir")
    ap.add_argument("--eval-shift", type=int, default=30,
                    help="iterations of a 2-parameter shift fitted per HELD-OUT crop with the "
                         "appearance frozen, for both the median texture and the fit; 0 = none")
    args = ap.parse_args()

    import smplx
    import torch

    from nfl_gsplat.compositing import fit_appearance as fa
    from nfl_gsplat.compositing import splat_torch as st
    from nfl_gsplat.compositing.appearance import (mask_turf, median_colours, turf_colour,
                                                   vertex_colours_from_view)
    from nfl_gsplat.compositing.mesh_to_gaussians import mesh_to_gaussians

    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=1)
    faces = model.faces.astype(np.int64)
    tracks_cam = load_camera_track(args.play_dir / "cameras.npz")
    cams = [c for c in ("sideline", "endzone") if c in tracks_cam]
    frames_of: dict[int, list[int]] = {}
    if args.from_timeline:
        from nfl_gsplat.render.play_timeline import load_play_timeline, placed_vertices

        tl, tracks_cam, _df, _frames_all, _poses = load_play_timeline(args.play_dir, model)
        # Every 6th frame, as the pose caches are; the timeline has all of them.
        state_of: dict[tuple[int, int], object] = {}
        for f in sorted(tl.states)[::6]:
            for s in tl.states[f]:
                frames_of.setdefault(s.pid, []).append(f)
                state_of[(s.pid, f)] = s
    else:
        blob = pickle.load(open(args.poses, "rb"))
        if not blob.get("world"):
            raise SetupError("05i needs the world-mode cache from 05f (transl in metres)")
        cache = blob["frames"]
        for f in sorted(cache):
            for pid in cache[f]:
                frames_of.setdefault(pid, []).append(f)
    bodies = [(pid, fs) for pid, fs in frames_of.items() if len(fs) >= MIN_FRAMES]
    if args.only_missing:
        bodies = [(pid, fs) for pid, fs in bodies
                  if not (args.out_dir / f"appearance_{pid}.npz").exists()]
    bodies.sort(key=lambda kv: -len(kv[1]))
    if args.limit:
        bodies = bodies[:args.limit]
    print(f"{len(bodies)} bodies with >= {MIN_FRAMES} posed frames; views {cams}")
    reader = FrameCache(args.play_dir, cams)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def placed(pid, f):
        if args.from_timeline:
            return placed_vertices(state_of[(pid, f)], model)
        rec = cache[f][pid]
        with torch.no_grad():
            betas = np.asarray(rec["betas"], np.float32)[None, :model.num_betas]
            body_pose = np.asarray(rec["body_pose"], np.float32).reshape(1, -1)
            orient = np.asarray(rec["global_orient"], np.float32).reshape(1, 3)
            res = model(betas=torch.tensor(betas), body_pose=torch.tensor(body_pose),
                        global_orient=torch.tensor(orient))
        return res.vertices[0].numpy().astype(np.float64) + np.asarray(rec["transl"], float)

    def crop_l1(colour, obs):
        """Held-out L1 of a body with per-vertex ``colour`` over ``obs``. With
        --eval-shift, each crop first gets a 2-parameter principal-point shift
        fitted with the appearance frozen, so placement error is charged
        equally to whatever appearance is being judged."""
        vals = []
        for ob in obs:
            batch = mesh_to_gaussians(ob.vertices, faces, colour=colour)
            scene = st.SceneParams.from_batch(batch, device=args.device)
            crop = fa.crop_for(ob, margin_px=6)
            x0, y0, w, h = crop
            target = torch.as_tensor(ob.image[y0:y0 + h, x0:x0 + w], device=args.device)
            K = torch.as_tensor(np.asarray(ob.K, np.float32), device=args.device)
            if args.eval_shift:
                shift = torch.zeros(2, device=args.device, requires_grad=True)
                opt = torch.optim.Adam([shift], lr=0.3)
                for _ in range(args.eval_shift):
                    opt.zero_grad(set_to_none=True)
                    Kf = K + torch.zeros_like(K).index_put(
                        (torch.tensor([0, 1], device=args.device),
                         torch.tensor([2, 2], device=args.device)), shift)
                    img = st.render(scene, Kf, ob.R, ob.t, crop=crop, background=target)
                    loss = (img - target).abs().mean()
                    if not loss.requires_grad:
                        break                        # nothing of the body in this crop
                    loss.backward()
                    opt.step()
                K = (K + torch.zeros_like(K).index_put(
                    (torch.tensor([0, 1], device=args.device),
                     torch.tensor([2, 2], device=args.device)), shift.detach()))
            with torch.no_grad():
                img = st.render(scene, K, ob.R, ob.t, crop=crop, background=target)
            vals.append((img - target).abs().mean().item())
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    t0 = time.time()
    for bi, (pid, fs) in enumerate(bodies):
        obs_train, obs_test = [], []
        samples = []
        for i, f in enumerate(fs):
            verts = placed(pid, f)
            for cam in cams:
                tr = tracks_cam[cam]
                if tr.conf[f] <= 0:
                    continue
                intr, pose = tr.at(f)
                K, R, t = intr.K(), pose.R, pose.t
                img = reader.get(cam, f)
                ob = fa.FrameObs(image=img, K=K, R=R, t=t, vertices=verts)
                # A body outside the frame, or clipped to a sliver, renders to
                # background with no gradient and crashes the fit.
                q = (K @ (R @ verts.T + t.reshape(3, 1))).T
                inside = (q[:, 2] > 0.1)
                uv = q[inside, :2] / np.maximum(q[inside, 2:], 1e-9)
                frac_in = float(((uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) &
                                 (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])).mean()) if len(uv) else 0.0
                cw, ch = fa.crop_for(ob, margin_px=6)[2:]
                if frac_in < 0.5 or cw < 16 or ch < 16:
                    continue
                (obs_test if (args.holdout and i % args.holdout == 0) else obs_train).append(ob)
                if not (args.holdout and i % args.holdout == 0):
                    sample = vertex_colours_from_view(verts, faces, K, R, t, img,
                                                      min_facing=args.min_facing)
                    if args.turf_mask:
                        sample = mask_turf(sample, turf_colour(img), dist=TURF_DIST)
                    samples.append(sample.astype(np.float16))
        if len(obs_train) < 4 or not obs_test:
            continue
        stack = np.stack(samples).astype(np.float32)
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
        valid = np.isfinite(med).all(1)
        # The body's dominant non-turf colour dresses the vertices that never
        # sampled anything but grass (thin limbs, the far side).
        body_colour = np.median(med[valid], axis=0) if valid.sum() >= 50 else np.asarray(BODY_RGB)
        colour0, _unseen = median_colours(med[None], fallback=body_colour)
        colour0 = np.asarray(colour0, np.float32)
        turf_ref = turf_colour(obs_train[0].image)
        turf_like0 = float((np.linalg.norm(colour0 - turf_ref[None], axis=1) < TURF_DIST).mean())
        l1_median = crop_l1(colour0, obs_test)
        cfg = fa.FitConfig(iters=args.iters, lr=args.lr, tv_weight=args.tv_weight,
                           device=args.device, log_every=0, translation=args.translation)
        fit, hist = fa.fit_body(colour0, faces, obs_train, cfg)
        fitted = fit.colour.cpu().numpy()
        l1_fit = crop_l1(fitted, obs_test)
        seen = fa.seen_vertices(faces, obs_train)
        unseen_colour = fitted[~seen].mean(0) if (~seen).any() else np.full(3, np.nan)
        # Turf bleed on the SEEN vertices: how green the body is. Turf is
        # (g - max(r, b)) well above zero; a jersey is not.
        c0, c1 = colour0[seen], fitted[seen]
        green0 = float(np.mean(np.clip(c0[:, 1] - c0[:, [0, 2]].max(1), 0, 1)))
        green1 = float(np.mean(np.clip(c1[:, 1] - c1[:, [0, 2]].max(1), 0, 1)))
        turf_like1 = float((np.linalg.norm(fitted - turf_ref[None], axis=1) < TURF_DIST).mean())
        mean_rgb = np.round(fitted[seen].mean(0), 2)
        np.savez(args.out_dir / f"appearance_{pid}.npz", colour=fitted,
                 log_scale_mult=fit.log_scale_mult.cpu().numpy(),
                 opacity_logit=fit.opacity_logit.cpu().numpy(),
                 l1_median=l1_median, l1_fit=l1_fit)
        summary[int(pid)] = {"frames": len(fs), "train": len(obs_train), "test": len(obs_test),
                             "l1_median": l1_median, "l1_fit": l1_fit,
                             "loss_first": hist["loss"][0], "loss_last": hist["loss"][-1],
                             "greenness_median": green0, "greenness_fit": green1,
                             "unseen_rgb": [float(x) for x in unseen_colour]}
        print(f"[{bi + 1}/{len(bodies)}] player {pid}: {len(obs_train)} train / {len(obs_test)} held-out "
              f"crops; train loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}; held-out L1 "
              f"median-texture {l1_median:.4f} -> fitted {l1_fit:.4f} "
              f"({100 * (1 - l1_fit / max(l1_median, 1e-9)):+.0f}%); greenness "
              f"{green0:.3f} -> {green1:.3f}; turf-like {turf_like0:.2f} -> {turf_like1:.2f}; "
              f"mean rgb {mean_rgb}; unseen rgb {np.round(unseen_colour, 2)}; "
              f"{time.time() - t0:.0f} s")
    if not summary:
        raise SetupError("no body had enough frames in both the train and held-out sets")
    gains = np.array([1 - v["l1_fit"] / max(v["l1_median"], 1e-9) for v in summary.values()])
    print(f"\nheld-out L1 gain over the median texture: median {100 * np.median(gains):+.0f}%, "
          f"worst {100 * gains.min():+.0f}%, bodies improved {int((gains > 0).sum())}/{len(gains)}")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
