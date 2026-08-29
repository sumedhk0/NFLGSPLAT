#!/usr/bin/env python
"""
Download synced Sideline + Endzone All-22 clips from NFL Pro (pro.nfl.com/film).

WHY THIS DESIGN
---------------
NFL Pro video pipeline:
    1. pro.nfl.com/api/secured/videos/filmroom/plays   -> play list (no video ids)
    2. pro.nfl.com/api/secured/videos/coaches          -> per-play All-22 asset ids (mcpID)
    3. POST api.nfl.com/play/v1/asset/{mcpID}           -> response.accessUrl = HLS master .m3u8
    4. fetch accessUrl (Anvato/Lura CDN)               -> variants + segments

Two facts:
  * The NFL bearer rotates and is bound to page-bootstrap cookies, so replaying the
    /secured/* calls externally gets 401s within a minute. => piggyback the logged-in browser.
  * accessUrl is self-signed (anvauth token, ~17 min TTL) and downloads with plain
    yt-dlp/ffmpeg, NO NFL auth.

HOW IT WORKS
------------
We connect to your already-logged-in Chrome over the DevTools Protocol and INJECT a
tiny fetch hook into the film page. That hook (running in the page, using the app's own
authenticated session) records every `accessUrl` the player mints, with its angle. Python
polls the hook and downloads each new clip with yt-dlp immediately (before the TTL lapses).

We inject in-page rather than using Playwright's network events because, over CDP against
a pre-existing tab, Playwright often cannot read cross-origin response bodies. In-page
hooking always can.

USAGE
-----
1) Launch a SECOND Chrome on a dedicated profile. Your normal Chrome may stay
   open -- a distinct --user-data-dir is an independent browser process, and it
   is only the DEFAULT profile that silently ignores the debugging port:
     & "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" `
         --remote-debugging-port=9222 --user-data-dir="C:\\Users\\sumedh\\chrome-nfl-profile"
2) Log into pro.nfl.com in it; open a game's film room:
     https://pro.nfl.com/film/plays?season=2025&seasonType=REG&weekSlug=WEEK_4&gameId=<FAPI_GAME_ID>
3) python scripts/06_download_all22.py --out data/all22/sea_at_az_wk4
4) In the browser step through plays and toggle Sideline/Endzone. Clips download automatically.

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

# In-page hook: records {key, mcpID, view, title, accessUrl} for every asset call.
# Idempotent (won't double-install); re-runs after navigations restore it.
INJECT_JS = r"""
() => {
  if (window.__all22) return window.__all22.length;
  window.__all22 = [];
  const bodies = new Map();
  const of = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url);
    const m = url && url.match(/api\.nfl\.com\/play\/v1\/asset\/(\d+)/);
    if (m && init && init.method === "POST") {
      try { bodies.set(url, (JSON.parse(init.body).asset) || {}); } catch (e) {}
    }
    const p = of.apply(this, arguments);
    if (m) {
      p.then(r => r.clone().json().then(j => {
        if (!j || !j.accessUrl) return;
        const a = bodies.get(url) || {};
        const key = m[1] + ":" + (a.videoView || "?");
        if (window.__all22.some(x => x.key === key)) return;
        window.__all22.push({ key, mcpID: m[1], view: a.videoView || "unknown",
                              title: a.title || m[1], accessUrl: j.accessUrl });
      }).catch(() => {})).catch(() => {});
    }
    return p;
  };
  return 0;
}
"""


def sanitize(text: str, maxlen: int = 70) -> str:
    text = re.sub(r"[^\w\s.-]", "", text or "").strip()
    return (re.sub(r"\s+", "_", text)[:maxlen]) or "play"


def download(idx: int, entry: dict, out_dir: Path, quality: str, manifest: Path) -> None:
    view = entry.get("view", "?")
    title = entry.get("title", entry.get("mcpID", ""))
    name = f"{idx:03d}_{view}_{sanitize(title)}"
    target = out_dir / f"{name}.mp4"
    print(f"[{idx:03d}] {view:8s} mcp={entry.get('mcpID')}  {title[:58]}")
    fmt = "bv*+ba/b" if quality == "best" else quality
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-part",
           "-f", fmt, "--remux-video", "mp4", "-o", str(target), entry["accessUrl"]]
    rc = subprocess.run(cmd).returncode
    ok = rc == 0 and target.exists()
    print(f"      -> {target.name}  [{'ok' if ok else f'FAIL(rc={rc})'}]")
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"index": idx, "mcpID": entry.get("mcpID"), "view": view,
                            "title": title, "file": target.name, "ok": ok}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/all22")
    ap.add_argument("--cdp", default="http://localhost:9222")
    ap.add_argument("--quality", default="best",
                    help="yt-dlp -f format ('best' or e.g. 'bv[height<=720]+ba/b')")
    ap.add_argument("--minutes", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    with sync_playwright() as p:
        print(f"Connecting to Chrome at {args.cdp} ...")
        browser = p.chromium.connect_over_cdp(args.cdp)

        def film_pages():
            pages = []
            for c in browser.contexts:
                for pg in c.pages:
                    if "pro.nfl.com" in (pg.url or ""):
                        pages.append(pg)
            return pages

        pages = film_pages()
        if not pages:
            print("No pro.nfl.com tab found. Open the film room in the debugged Chrome, then rerun.")
            return
        for pg in pages:
            print(f"Watching: {pg.url}")

        seen: set[str] = set()
        idx = 0
        hb = 0
        deadline = time.time() + args.minutes * 60
        print(f"\nHarvesting {args.minutes} min. Step through plays + toggle Sideline/Endzone.\n")
        try:
            while time.time() < deadline:
                for pg in film_pages():
                    try:
                        pg.evaluate(INJECT_JS)                       # (re)install hook
                        entries = pg.evaluate("window.__all22 || []")  # read captures
                    except Exception:
                        continue
                    for e in entries:
                        key = e.get("key")
                        if key and key not in seen and e.get("accessUrl"):
                            seen.add(key)
                            idx += 1
                            download(idx, e, out_dir, args.quality, manifest)
                hb += 1
                if hb % 15 == 0:  # ~ every 30s
                    print(f"...watching ({len(seen)} captured so far). "
                          f"Toggle Sideline/Endzone on a play to trigger capture.")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
    print(f"\nDone. {len(seen)} clips. Manifest: {manifest}")


if __name__ == "__main__":
    main()
