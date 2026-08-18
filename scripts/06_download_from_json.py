#!/usr/bin/env python
"""
Download All-22 clips from a urls.json produced by 06_harvest_console_snippet.js.

    python scripts/06_download_from_json.py urls.json --out data/all22

Each entry: {"mcpID","view","title","accessUrl"}. accessUrls expire ~17 min after
capture, so run this promptly after harvesting. Requires yt-dlp + ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def sanitize(text: str, maxlen: int = 70) -> str:
    text = re.sub(r"[^\w\s.-]", "", text or "").strip()
    return (re.sub(r"\s+", "_", text)[:maxlen]) or "play"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="urls.json from the console harvester")
    ap.add_argument("--out", default="data/all22")
    ap.add_argument("--quality", default="bv*+ba/b", help="yt-dlp -f format string")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.json).read_text(encoding="utf-8"))
    print(f"{len(entries)} clips to download -> {out}")

    for i, e in enumerate(entries, 1):
        name = f"{i:03d}_{e.get('view','?')}_{sanitize(e.get('title',''))}.mp4"
        target = out / name
        print(f"[{i:03d}] {e.get('view'):8s} {e.get('title','')[:55]}")
        cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-part",
               "-f", args.quality, "--remux-video", "mp4", "-o", str(target), e["accessUrl"]]
        rc = subprocess.run(cmd).returncode
        print(f"      -> {name}  [{'ok' if rc == 0 and target.exists() else f'FAIL rc={rc}'}]")


if __name__ == "__main__":
    main()
