#!/usr/bin/env python
r"""Download EVERY play of a game -- both angles -- from NFL Pro, without clicking.

``06_download_all22.py`` hooks the page's ``fetch`` and waits for a human to step
through the play list toggling Sideline and Endzone. That is fine for one play
and unusable for a game, let alone a season.

This drives the app's OWN APIs from inside the logged-in page, so it inherits the
session instead of replaying it:

    1. /api/secured/videos/filmroom/plays  -> the plays in a game
    2. /api/secured/videos/coaches         -> per-play asset ids (mcpID) per angle
    3. POST api.nfl.com/play/v1/asset/{id} -> a signed HLS accessUrl
    4. yt-dlp fetches the accessUrl        -> no NFL auth needed at this step

WHY IN-PAGE. The NFL bearer rotates and is bound to page-bootstrap cookies, so
replaying /secured/* from Python gets 401s within a minute. Do not reimplement
the auth; borrow it. Nothing here reads, stores or logs a token.

WHY IN BATCHES. accessUrl is self-signed with roughly a 17-minute TTL. Minting
every URL for a 150-play game up front would leave the last ones dead before
yt-dlp reached them, and the failure looks like a network error rather than an
expiry. So a batch is minted, downloaded, and only then is the next minted.

SETUP
    1) Launch a SECOND Chrome on a dedicated profile. Your normal Chrome can stay
       open: a distinct --user-data-dir starts an independent browser process.
       (Using your DEFAULT profile is what fails -- Chrome hands the request to
       the already-running process and silently ignores the debugging port.)
         & "C:\Program Files\Google\Chrome\Application\chrome.exe" `
             --remote-debugging-port=9222 --user-data-dir="C:\Users\sumedh\chrome-nfl-profile"
    2) Log into pro.nfl.com in THAT window -- once; the profile persists it --
       and open a film-room page, e.g.
         https://pro.nfl.com/film/plays?season=2024&seasonType=REG&weekSlug=WEEK_5
    3) python scripts/06c_download_game_auto.py --season 2024 --week 5 \
             --game-id <FAPI_GAME_ID> --out data/all22/<name>

    --list-games prints the games for a week if you do not have the id.

Requires: playwright, yt-dlp, ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Mint at most this many signed URLs before downloading them. Well inside the
# ~17 minute TTL even when a clip takes a minute to pull.
BATCH = 6

# Ask the page for the plays in a game, then for the coaches-film assets, and
# hand back every (mcpID, view) pair found. The response schemas are walked
# GENERICALLY rather than by a fixed path: these are private endpoints whose
# shape is not contracted with us, and a rename would otherwise fail with a
# KeyError deep in a comprehension instead of an actionable message.
LIST_ASSETS_JS = r"""
async ({season, seasonType, week, gameId}) => {
  const out = {plays: [], assets: [], errors: []};
  const get = async (url) => {
    const r = await fetch(url, {credentials: "include"});
    if (!r.ok) { out.errors.push(url + " -> HTTP " + r.status); return null; }
    return r.json();
  };

  // Walk any JSON and collect objects that look like a film asset.
  const walk = (node, hit) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) { node.forEach(n => walk(n, hit)); return; }
    hit(node);
    for (const k of Object.keys(node)) walk(node[k], hit);
  };

  const qs = `season=${season}&seasonType=${seasonType}&week=${week}` +
             (gameId ? `&gameId=${gameId}` : "");
  const plays = await get(`/api/secured/videos/filmroom/plays?${qs}`);
  if (plays) {
    walk(plays, n => {
      if (n.playId != null && (n.gameId != null || n.gsisId != null)) {
        out.plays.push({playId: n.playId, gameId: n.gameId ?? n.gsisId ?? null,
                        title: n.playDescription ?? n.title ?? null});
      }
    });
  }
  const coaches = await get(`/api/secured/videos/coaches?${qs}`);
  for (const src of [coaches, plays]) {
    if (!src) continue;
    walk(src, n => {
      const id = n.mcpID ?? n.mcpId ?? n.assetId;
      const view = n.videoView ?? n.view ?? n.angle;
      if (id && view) {
        out.assets.push({mcpID: String(id), view: String(view),
                         title: String(n.title ?? n.playDescription ?? id),
                         playId: n.playId ?? null});
      }
    });
  }
  // de-duplicate on (mcpID, view)
  const seen = new Set();
  out.assets = out.assets.filter(a => {
    const k = a.mcpID + ":" + a.view;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return out;
}
"""

# Mint the signed URL for one asset, exactly as the player does.
MINT_JS = r"""
async ({mcpID, view, title}) => {
  const r = await fetch(`https://api.nfl.com/play/v1/asset/${mcpID}`, {
    method: "POST",
    credentials: "include",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({asset: {videoView: view, title: title}}),
  });
  if (!r.ok) return {error: "HTTP " + r.status};
  const j = await r.json();
  return j && j.accessUrl ? {accessUrl: j.accessUrl} : {error: "no accessUrl"};
}
"""


def sanitize(text: str, maxlen: int = 70) -> str:
    text = re.sub(r"[^\w\s.-]", "", text or "").strip()
    return (re.sub(r"\s+", "_", text)[:maxlen]) or "play"


def download(entry: dict, access_url: str, out_dir: Path, quality: str,
             manifest: Path, index: int) -> bool:
    view = entry.get("view", "?")
    title = entry.get("title", entry["mcpID"])
    target = out_dir / f"{index:03d}_{view}_{sanitize(title)}.mp4"
    if target.exists() and target.stat().st_size > 0:
        print(f"[{index:03d}] {view:9s} already have {target.name}")
        return True
    fmt = "bv*+ba/b" if quality == "best" else quality
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-part",
           "-f", fmt, "--remux-video", "mp4", "-o", str(target), access_url]
    rc = subprocess.run(cmd).returncode
    ok = rc == 0 and target.exists()
    # The accessUrl carries a signed token; it is deliberately NOT logged here
    # or written to the manifest.
    print(f"[{index:03d}] {view:9s} {title[:52]:52s} "
          f"{'ok' if ok else f'FAIL rc={rc}'}")
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"index": index, "mcpID": entry["mcpID"],
                             "view": view, "title": title,
                             "playId": entry.get("playId"),
                             "file": target.name, "ok": ok}) + "\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season-type", default="REG")
    ap.add_argument("--game-id", default=None,
                    help="FAPI game id; omit with --list-games to discover it")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cdp", default="http://localhost:9222")
    ap.add_argument("--quality", default="best")
    ap.add_argument("--views", default="Sideline,Endzone",
                    help="comma-separated angles to keep")
    ap.add_argument("--limit", type=int, default=0, help="stop after N plays")
    ap.add_argument("--list-games", action="store_true",
                    help="print what the page reports and exit")
    args = ap.parse_args()

    wanted_views = {v.strip().lower() for v in args.views.split(",") if v.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(args.cdp)
        except Exception as exc:
            raise SystemExit(
                f"cannot reach Chrome at {args.cdp} ({exc}).\n"
                "Close ALL Chrome windows and relaunch with:\n"
                '  chrome.exe --remote-debugging-port=9222 '
                '--user-data-dir="C:\\Users\\sumedh\\chrome-nfl-profile"\n'
                "then log in to pro.nfl.com in that window.") from exc

        pages = [p for ctx in browser.contexts for p in ctx.pages]
        page = next((p for p in pages if "pro.nfl.com" in (p.url or "")), None)
        if page is None:
            raise SystemExit(
                "no pro.nfl.com tab is open in that Chrome. Open a film-room "
                "page there first -- the session lives in the page, and these "
                "endpoints refuse a replayed bearer.")
        print(f"attached to: {page.url[:90]}")

        found = page.evaluate(LIST_ASSETS_JS, {
            "season": args.season, "seasonType": args.season_type,
            "week": args.week, "gameId": args.game_id})

        for err in found.get("errors", []):
            print(f"  ! {err}")
        plays, assets = found.get("plays", []), found.get("assets", [])
        print(f"plays reported: {len(plays)};  film assets found: {len(assets)}")
        if args.list_games or not assets:
            seen = {}
            for p in plays:
                seen.setdefault(p.get("gameId"), 0)
                seen[p["gameId"]] += 1
            for gid, n in sorted(seen.items(), key=lambda kv: -kv[1])[:32]:
                print(f"   gameId {gid}: {n} plays")
            if not assets:
                raise SystemExit(
                    "no film assets came back. Either the game id is wrong, or "
                    "these endpoints have been renamed -- rerun with "
                    "--list-games and check what the page returns.")
            return

        keep = [a for a in assets if a["view"].lower() in wanted_views]
        print(f"keeping {len(keep)} of {len(assets)} assets "
              f"({sorted({a['view'] for a in assets})})")
        by_play: dict = {}
        for a in keep:
            by_play.setdefault(a.get("playId") or a["mcpID"], []).append(a)
        ordered = [a for _pid, group in sorted(by_play.items(), key=lambda kv: str(kv[0]))
                   for a in group]
        if args.limit:
            ordered = ordered[:args.limit * len(wanted_views)]

        print(f"downloading {len(ordered)} clips in batches of {BATCH} "
              f"(signed URLs expire in ~17 min, so they are minted per batch)\n")
        n_ok = 0
        for start in range(0, len(ordered), BATCH):
            batch = ordered[start:start + BATCH]
            minted = []
            for entry in batch:
                res = page.evaluate(MINT_JS, {"mcpID": entry["mcpID"],
                                              "view": entry["view"],
                                              "title": entry["title"]})
                if res.get("accessUrl"):
                    minted.append((entry, res["accessUrl"]))
                else:
                    print(f"      mint failed for {entry['mcpID']} "
                          f"{entry['view']}: {res.get('error')}")
            for offset, (entry, url) in enumerate(minted):
                n_ok += download(entry, url, args.out, args.quality, manifest,
                                 start + offset + 1)
            time.sleep(0.5)

        print(f"\n{n_ok}/{len(ordered)} clips downloaded -> {args.out}")
        print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
