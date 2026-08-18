// ---------------------------------------------------------------------------
// NFL Pro All-22 accessUrl harvester — paste into the pro.nfl.com/film DevTools
// console. Zero-setup alternative to the Playwright script for a handful of plays.
//
// After pasting, step through plays in the UI and toggle Sideline / Endzone for
// each one you want. Every clip the player loads is captured. Then run:
//     copy(window.__all22dump())      // copies JSON to your clipboard
// and paste it into a file (e.g. urls.json), then feed it to a yt-dlp downloader.
//
// NOTE: each accessUrl is a signed URL valid ~17 minutes. Download promptly.
// ---------------------------------------------------------------------------
(() => {
  if (window.__all22) { console.log("already installed;", window.__all22.length, "captured"); return; }
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
        if (!j.accessUrl) return;
        const a = bodies.get(url) || {};
        const key = m[1] + ":" + (a.videoView || "?");
        if (window.__all22.some(x => x.key === key)) return;
        window.__all22.push({ key, mcpID: m[1], view: a.videoView || "unknown",
                              title: a.title || m[1], accessUrl: j.accessUrl });
        console.log("captured", window.__all22.length, a.videoView, (a.title || "").slice(0, 50));
      }).catch(() => {})).catch(() => {});
    }
    return p;
  };
  window.__all22dump = () => JSON.stringify(window.__all22, null, 1);
  console.log("Installed. Step through plays + toggle Sideline/Endzone. Then: copy(window.__all22dump())");
})();
