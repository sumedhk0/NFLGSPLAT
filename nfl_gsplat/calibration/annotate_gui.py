"""Interactive OpenCV GUI: click NFL field landmarks on a reference frame.

Usage from the CLI (invoked by ``scripts/02_calibrate_cameras.py``):

    annotate(video_path, out_json, width_height, preset_names=None)

Controls (shown on a HUD overlay):
    mouse click    — place point for the currently highlighted landmark
    n / p          — next / previous landmark in the list
    d              — delete the currently selected landmark's placement
    u              — undo last placement
    s              — save to JSON and exit
    q              — quit without saving
    z              — toggle zoom lens at cursor

The saved JSON is a list of ``{"name", "uv", "frame"}`` entries matching
what :mod:`solve_pnp` expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from nfl_gsplat.calibration.field_landmarks import list_landmark_names
from nfl_gsplat.utils.io import write_json

# A short default preset — a reasonable minimum for a sideline view. Users
# should click as many as are visible; more is better.
DEFAULT_PRESET: list[str] = [
    "mid_50_left_sideline",  "mid_50_right_sideline",
    "mid_50_left_hash",      "mid_50_right_hash",
    "home_25_left_sideline", "home_25_right_sideline",
    "home_25_left_hash",     "home_25_right_hash",
    "away_25_left_sideline", "away_25_right_sideline",
    "away_25_left_hash",     "away_25_right_hash",
    "home_goal_left_sideline", "home_goal_right_sideline",
    "away_goal_left_sideline", "away_goal_right_sideline",
]


def _grab_frame(video_path: Path | str, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
    return frame


def _draw_hud(
    img: np.ndarray,
    current_name: str,
    placed: dict[str, tuple[float, float]],
    remaining: list[str],
) -> np.ndarray:
    out = img.copy()
    # Placed markers.
    for name, (u, v) in placed.items():
        color = (0, 255, 255) if name == current_name else (0, 200, 0)
        cv2.circle(out, (int(u), int(v)), 6, color, 2)
        cv2.putText(out, name, (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # Top banner.
    banner = f"CURRENT: {current_name}    placed={len(placed)}    remaining={len(remaining)}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, banner, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # Bottom help.
    help_text = "click: place   n/p: next/prev   u: undo   d: delete   s: save   q: quit   z: zoom"
    cv2.rectangle(out, (0, out.shape[0] - 26), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, help_text, (8, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def annotate_frame(
    img: np.ndarray,
    names: list[str],
    prefill: dict[str, tuple[float, float]] | None = None,
    window_title: str = "NFL landmark annotator",
) -> list[tuple[str, tuple[float, float]]]:
    """Run the single-frame click loop and return the placed points.

    Parameters
    ----------
    img:
        BGR image to annotate (not modified).
    names:
        Ordered list of landmark names the user can place.
    prefill:
        Optional ``name → (u, v)`` dict to pre-seed placed points.
    window_title:
        OpenCV window title.

    Returns
    -------
    list of ``(name, (u, v))`` tuples in placement order.
    """
    name_list = list(names)
    idx = 0
    placed: dict[str, tuple[float, float]] = dict(prefill) if prefill else {}
    history: list[str] = list(placed.keys())

    def _on_mouse(event, x, y, flags, _):
        nonlocal idx
        if event == cv2.EVENT_LBUTTONDOWN:
            placed[name_list[idx]] = (float(x), float(y))
            history.append(name_list[idx])
            # Advance to next un-placed landmark.
            for k in range(1, len(name_list) + 1):
                cand = (idx + k) % len(name_list)
                if name_list[cand] not in placed:
                    idx = cand
                    break

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    # Realize the window before attaching the mouse callback. Some OpenCV Qt builds
    # (notably over VNC / X-forwarding) don't create the underlying window until the
    # first imshow, so setMouseCallback right after namedWindow fails with a
    # "NULL window handler". A throwaway imshow+waitKey forces the window to exist.
    cv2.imshow(window_title, img)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_title, _on_mouse)

    while True:
        remaining = [n for n in name_list if n not in placed]
        view = _draw_hud(img, name_list[idx], placed, remaining)
        cv2.imshow(window_title, view)
        # If the window was closed (WM X button / stray key), don't hang or lose
        # the current frame — treat it as save & continue (return placed points).
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise RuntimeError("annotation aborted by user")
        if key == ord("n"):
            idx = (idx + 1) % len(name_list)
        elif key == ord("p"):
            idx = (idx - 1) % len(name_list)
        elif key == ord("u"):
            if history:
                placed.pop(history.pop(), None)
        elif key == ord("d"):
            placed.pop(name_list[idx], None)
        elif key == ord("s"):
            break

    cv2.destroyAllWindows()
    return [(n, placed[n]) for n in name_list if n in placed]


def annotate(
    video_path: Path | str,
    out_json: Path | str,
    *,
    frame_index: int = 0,
    preset_names: Sequence[str] | None = None,
    window_title: str = "NFL landmark annotator",
) -> Path:
    """Open the GUI and write ``out_json`` on save. Returns the output path.

    The list of candidate landmark names defaults to :data:`DEFAULT_PRESET`.
    Users can cycle through ``list_landmark_names()`` with ``n`` / ``p`` if
    a particular landmark isn't in the preset.
    """
    img = _grab_frame(video_path, frame_index=frame_index)
    all_names = list_landmark_names()
    name_list: list[str] = list(preset_names) if preset_names else list(DEFAULT_PRESET)
    # Ensure every preset name is valid.
    for n in name_list:
        if n not in all_names:
            raise ValueError(f"preset contains unknown landmark {n!r}")

    result = annotate_frame(img, name_list, window_title=window_title)
    entries = [
        {"name": name, "uv": [u, v], "frame": int(frame_index)}
        for name, (u, v) in result
    ]
    write_json(out_json, entries)
    return Path(out_json)
