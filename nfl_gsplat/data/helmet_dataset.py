"""Read the NFL Helmet Assignment set into this project's conventions.

The set supplies what nothing else here has: exactly 22 players per play, for
the whole play, each labelled with team and jersey, seen from BOTH cameras, with
per-frame helmet boxes already assigned to a player.

Two conversions are needed, and both are places to get quietly wrong.

FIELD FRAME. The tracking uses the NFL's own convention -- x from 0 to 120
yards INCLUDING both end zones, y from 0 to 53 1/3 yards from one sideline --
while everything in this repo is metres centred on midfield, x toward the end
lines and y across. Mixing the two produces positions that look plausible (they
land on a field) and are 50 metres wrong.

TIME. Tracking is 10 Hz; the video is 59.94 fps. They are aligned on the
``ball_snap`` event rather than on the first sample, because the clips do not
start at a fixed offset from the snap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.field_landmarks import YARD_TO_M
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# The NFL tracking frame: 120 yd long counting both end zones, so midfield is
# x = 60, and 53 1/3 yd wide, so the centre line is y = 80/3.
NFL_MIDFIELD_X_YD: float = 60.0
NFL_MID_Y_YD: float = 160.0 / 3.0 / 2.0        # 26.667

TRACKING_HZ: float = 10.0
SNAP_EVENT: str = "ball_snap"


@dataclass(frozen=True)
class PlayTracking:
    """One play's ground truth, in FIELD METRES."""

    game_key: int
    play_id: int
    players: tuple[str, ...]                    # "H97", "V23", ...
    times: np.ndarray                           # [T] seconds relative to the snap
    xy: np.ndarray                              # [T, P, 2] metres, midfield origin
    snap_time: float | None                     # seconds into the clip, if known

    @property
    def n_players(self) -> int:
        return len(self.players)

    def at(self, seconds_from_snap: float) -> np.ndarray:
        """Positions at a time, linearly interpolated. ``[P, 2]`` metres."""
        idx = np.clip(np.searchsorted(self.times, seconds_from_snap), 1,
                      len(self.times) - 1)
        t0, t1 = self.times[idx - 1], self.times[idx]
        w = 0.0 if t1 == t0 else (seconds_from_snap - t0) / (t1 - t0)
        w = float(np.clip(w, 0.0, 1.0))
        return (1.0 - w) * self.xy[idx - 1] + w * self.xy[idx]


def to_field_metres(x_yd, y_yd):
    """NFL tracking yards -> this project's metres, midfield at the origin.

    x runs 0..120 with the end zones included, so 60 is midfield; y runs
    0..53 1/3 across, so 80/3 is the centre line.
    """
    x = (np.asarray(x_yd, float) - NFL_MIDFIELD_X_YD) * YARD_TO_M
    y = (np.asarray(y_yd, float) - NFL_MID_Y_YD) * YARD_TO_M
    return x, y


def load_tracking(path: Path | str):
    """``{(game_key, play_id): PlayTracking}`` from train_player_tracking.csv."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"tracking not found at {path}. Run "
            "scripts/07_fetch_helmet_dataset.py first.")
    df = pd.read_csv(path)
    need = {"gameKey", "playID", "player", "time", "x", "y"}
    missing = need - set(df.columns)
    if missing:
        raise SetupError(f"{path} is missing columns {sorted(missing)}")

    # Timezone-NAIVE datetime64 on purpose. A tz-aware column comes back from
    # .unique() as object dtype, and numpy then refuses to subtract it with a
    # message about ufunc operand types that says nothing about timezones.
    df["time"] = (pd.to_datetime(df["time"], format="mixed", utc=True)
                    .dt.tz_localize(None))
    out: dict[tuple[int, int], PlayTracking] = {}
    for (game, play), grp in df.groupby(["gameKey", "playID"]):
        players = tuple(sorted(grp["player"].unique()))
        stamps = np.sort(grp["time"].drop_duplicates().to_numpy())
        # Anchor on the snap. Falling back to the clip start would shift every
        # play by a different amount, since the clips are not cut at a fixed
        # offset from it.
        if "event" in grp.columns:
            snapped = grp.loc[grp["event"] == SNAP_EVENT, "time"]
        else:
            snapped = grp.iloc[0:0]["time"]
        origin = snapped.min().to_datetime64() if len(snapped) else stamps[0]
        secs = (stamps - origin) / np.timedelta64(1, "s")

        index = {p: i for i, p in enumerate(players)}
        xy = np.full((len(stamps), len(players), 2), np.nan)
        gx, gy = to_field_metres(grp["x"].to_numpy(), grp["y"].to_numpy())
        # Scatter by position rather than looking each timestamp up in a dict.
        # Iterating the column yields pandas Timestamps while ``stamps`` holds
        # numpy datetime64, so a dict keyed on one and probed with the other
        # raises a KeyError that reads like missing data.
        rows = np.searchsorted(stamps, grp["time"].to_numpy())
        cols = grp["player"].map(index).to_numpy()
        xy[rows, cols] = np.column_stack([gx, gy])
        out[(int(game), int(play))] = PlayTracking(
            game_key=int(game), play_id=int(play), players=players,
            times=secs.astype(float), xy=xy,
            snap_time=(float((origin - stamps[0]) / np.timedelta64(1, "s"))
                       if len(snapped) else None))

    counts = {k: v.n_players for k, v in out.items()}
    odd = {k: n for k, n in counts.items() if n != 22}
    _LOG.info("loaded %d plays; %d have exactly 22 players%s", len(out),
              sum(1 for n in counts.values() if n == 22),
              f", {len(odd)} do not" if odd else "")
    return out


def load_labels(path: Path | str):
    """Per-frame helmet boxes already assigned to a player.

    Returns the frame unchanged apart from a ``play_key`` column, because this
    is ground truth and reshaping it invites silent loss.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"labels not found at {path}. Run "
            "scripts/07_fetch_helmet_dataset.py first.")
    df = pd.read_csv(path)
    df["play_key"] = list(zip(df["gameKey"].astype(int), df["playID"].astype(int)))
    return df


def video_name(game_key: int, play_id: int, view: str) -> str:
    """The clip filename for a play and angle, e.g. 57583_000082_Endzone.mp4."""
    if view not in ("Endzone", "Sideline"):
        raise ValueError(f"view must be Endzone or Sideline, got {view!r}")
    return f"{int(game_key)}_{int(play_id):06d}_{view}.mp4"
