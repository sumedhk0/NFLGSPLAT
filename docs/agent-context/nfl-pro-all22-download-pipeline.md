---
name: nfl-pro-all22-download-pipeline
description: Reverse-engineered NFL Pro (pro.nfl.com/film) All-22 sideline+endzone video download pipeline + auth reality + working scripts
metadata: 
  node_type: memory
  type: reference
  originSessionId: bea9ca4d-e740-4d84-a477-e0fcb46df302
  modified: 2026-07-25T01:10:44.463Z
---

Reverse-engineered 2026-07-24 via Chrome extension network capture, on the user's own NFL+ Premium account (skothari67, "MEI"), for sourcing real endzone/sideline footage for the calibration project ([[endzone-camera-bringup]]).

**Video pipeline (pro.nfl.com film room):**
1. `GET pro.nfl.com/api/secured/videos/filmroom/plays?season&seasonType&weekSlug&gameId` → play list (playId, sequence, gameId numeric + fapiGameId, description). **No video ids.**
2. `GET pro.nfl.com/api/secured/videos/coaches?gameId&playId` → per-play All-22 asset ids (`mcpID`). (Shape not fully confirmed — 401'd on replay.)
3. `POST api.nfl.com/play/v1/asset/{mcpID}` with body `{asset:{videoView:"sideline"|"endzone", mcpID, title, ...}, deviceInfo(base64), device, videoId, ...}` → response JSON `{accessUrl, init, metadata, source:"lura"}`. **`accessUrl` = HLS master .m3u8** on `dcs-vod.mp.lura.live` (Anvato/Lura). Sideline & endzone are separate mcpIDs (e.g. first kickoff: sideline 2322980, endzone 2322999).
4. Fetch accessUrl → variant `prog.m3u8` → segments. No DRM observed; plain HLS.

**Auth reality (the load-bearing gotcha):**
- Bearer JWT lives at `localStorage["keystore.token"].accessToken` (~1h exp, has refreshToken + clientId/clientKey). Minted/refreshed via `api.nfl.com/identity/v3/token`; user login via OIDC `api.nfl.com/accounts/v1/auth/oidc/*`.
- The `/secured/*` endpoints (plays, coaches) are **cookie-bootstrap-bound + rotate**: replaying with the stored bearer returns 200 only in a short window right after page load, then 401s within ~1-2 min even though token not expired (811s left). Do NOT reimplement NFL auth.
- `api.nfl.com/play/v1/asset/{mcpID}` IS replayable in-page (got 200 by reusing captured headers+body). `{}` body → 403 "missing deviceId" (deviceId comes from the base64 `deviceInfo` field, not a header).
- **accessUrl is self-signed** (`anvauth` token, TTL ≈ 17 min: te−t ≈ 1032s). Downloads with plain yt-dlp/ffmpeg, **no NFL auth**. => split: browser mints accessUrl, yt-dlp downloads.

**Approach = piggyback the logged-in browser, intercept `api.nfl.com/play/v1/asset/*` responses, yt-dlp each accessUrl immediately.** Scripts written: `scripts/06_download_all22.py` (Playwright over CDP, --manual harvest+download), `scripts/06_harvest_console_snippet.js` (paste-in interceptor → `copy(window.__all22dump())`), `scripts/06_download_from_json.py`. Env has ffmpeg 8.1.2, yt-dlp 2026.03.17 (`python -m yt_dlp`), playwright importable.

**NOT yet done:** end-to-end download not run (needs user to relaunch Chrome with `--remote-debugging-port=9222`); coaches response shape unconfirmed; play-row DOM selector for auto-stepping unpinned (rows virtualized) — manual stepping used instead.
