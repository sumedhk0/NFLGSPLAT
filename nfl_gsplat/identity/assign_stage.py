"""Identity wiring stage: tracks → team/referee classification → entities.json.

Sits between cross-camera re-ID + jersey OCR and the pose/avatar stages. For
each cross-camera player (``global_player_id``) it:

1. takes a representative torso crop and runs :mod:`team_color`
   (team cluster + referee stripe test);
2. labels the two color clusters to real teams by **roster membership** of their
   OCR'd jerseys (no need for hand-specified team colors);
3. attaches ``team`` / ``is_referee`` columns and calls
   :func:`nfl_gsplat.identity.registry.resolve_tracks` (grouped by
   ``global_player_id``);
4. writes ``entities.json`` and updates the season registry.

``entities.json`` records, per on-field instance::

    {instance_id, player_uid, entity_type}

``instance_id`` (the global_player_id) keys the per-instance pose file; multiple
referees share ``player_uid == "__referee__"`` but keep distinct ``instance_id``s
so each is posed by its own motion.

The classification core takes pre-extracted crops so it is CPU-testable;
:func:`extract_representative_crops` (which reads the video) is the env-side glue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from nfl_gsplat.identity.registry import (
    REFEREE_UID,
    Assignment,
    EntityType,
    IdentityMatchConfig,
    assign_identities,
    resolve_tracks,
)
from nfl_gsplat.identity.roster import RosterEntry
from nfl_gsplat.identity.team_color import (
    RefereeConfig,
    dominant_jersey_color,
    is_referee,
    split_two_teams,
)
from nfl_gsplat.utils.io import write_json
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def _voted_jerseys(tracks_df: pd.DataFrame, id_col: str) -> dict:
    out: dict = {}
    for gid, grp in tracks_df.groupby(id_col):
        v = grp["jersey_number_ocr"]
        v = v[v >= 0]
        out[gid] = int(v.value_counts().idxmax()) if len(v) else -1
    return out


def classify_entities(
    crops_by_id: Mapping,
    jerseys_by_id: Mapping,
    candidates: list[RosterEntry],
    home_team: str,
    away_team: str,
    ref_cfg: RefereeConfig | None = None,
) -> dict:
    """Return ``{id: (team|None, is_referee)}``.

    Teams come from 2-means color clustering, with clusters labeled to real
    teams by which roster their members' jerseys match more often.
    """
    ids = list(crops_by_id.keys())
    if not ids:
        return {}
    refs = {g: bool(is_referee(crops_by_id[g], ref_cfg)) for g in ids}

    # Cluster team colors over *players only* — a referee's grayscale stripes
    # would otherwise pull a cluster center and flip the labels.
    player_ids = [g for g in ids if not refs[g]]
    label_by_id: dict = {}
    if player_ids:
        colors = np.stack([dominant_jersey_color(crops_by_id[g]) for g in player_ids])
        labels = split_two_teams(colors)
        label_by_id = {g: int(lab) for g, lab in zip(player_ids, labels)}

    home_j = {c.jersey for c in candidates if c.team == home_team}
    away_j = {c.jersey for c in candidates if c.team == away_team}

    # Per-cluster home-vs-away affinity from members' jerseys.
    margin = {0: 0, 1: 0}
    for g in player_ids:
        j = jerseys_by_id.get(g, -1)
        c = label_by_id[g]
        if j in home_j:
            margin[c] += 1
        if j in away_j:
            margin[c] -= 1

    if not candidates:
        cluster_team: dict = {0: None, 1: None}
    else:
        home_cluster = 0 if margin[0] >= margin[1] else 1
        cluster_team = {home_cluster: home_team, 1 - home_cluster: away_team}

    return {
        g: (None, True) if refs[g] else (cluster_team[label_by_id[g]], False)
        for g in ids
    }


def assign_play_identities(
    tracks_df: pd.DataFrame,
    crops_by_id: Mapping,
    candidates: list[RosterEntry],
    home_team: str,
    away_team: str,
    cfg: IdentityMatchConfig,
    *,
    id_col: str = "global_player_id",
    ref_cfg: RefereeConfig | None = None,
) -> tuple[pd.DataFrame, list[Assignment]]:
    """Classify teams/referees, attach columns, and resolve identities."""
    jerseys = _voted_jerseys(tracks_df, id_col)
    ent = classify_entities(crops_by_id, jerseys, candidates, home_team, away_team, ref_cfg)

    df = tracks_df.copy()
    df["team"] = df[id_col].map({g: t for g, (t, _) in ent.items()})
    df["is_referee"] = df[id_col].map({g: r for g, (_, r) in ent.items()}).fillna(False)

    resolved = resolve_tracks(df, candidates, cfg, id_col=id_col)
    assignments = assign_identities(df, candidates, cfg, id_col=id_col)
    return resolved, assignments


def write_entities_json(path: Path | str, assignments: list[Assignment]) -> Path:
    """Persist the renderable entities (players + referees; OTHER dropped).

    One entry per instance (``instance_id`` = the resolved id), so referees that
    share ``__referee__`` keep distinct pose files.
    """
    entities = [
        {"instance_id": str(a.track_id), "player_uid": a.player_uid, "entity_type": a.entity_type}
        for a in assignments
        if a.entity_type != EntityType.OTHER.value and a.player_uid
    ]
    write_json(path, entities)
    n_ref = sum(1 for e in entities if e["player_uid"] == REFEREE_UID)
    _LOG.info(f"entities.json: {len(entities)} instances ({n_ref} referee) → {path}")
    return Path(path)


def extract_representative_crops(
    tracks_df: pd.DataFrame,
    video_paths: Mapping[str, object],
    *,
    id_col: str = "global_player_id",
    torso_frac: float = 0.55,
) -> dict:
    """One representative torso crop per identity (the largest-bbox detection).

    Reads a single frame per identity (cheap) and crops the upper ``torso_frac``
    of the bbox — the jersey region :mod:`team_color` clusters on. Env-side glue
    (needs the video + ``cv2``); the classification core takes the crops it
    returns, so identity logic stays CPU-testable.
    """
    import cv2

    def _read(video, frame_idx):
        cap = cv2.VideoCapture(str(video))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, img = cap.read()
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None
        finally:
            cap.release()

    crops: dict = {}
    df = tracks_df.copy()
    df["_area"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])
    for gid, grp in df.groupby(id_col):
        r = grp.loc[grp["_area"].idxmax()]
        video = video_paths.get(str(r["cam"]))
        if video is None:
            continue
        frame = _read(video, int(r["frame"]))
        if frame is None:
            continue
        x1, y1 = max(0, int(r["bbox_x1"])), max(0, int(r["bbox_y1"]))
        x2, y2 = min(frame.shape[1], int(r["bbox_x2"])), min(frame.shape[0], int(r["bbox_y2"]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        y_torso = y1 + int(torso_frac * (y2 - y1))
        crops[gid] = frame[y1:y_torso, x1:x2]
    return crops


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    import pandas as pd_
    import typer

    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.identity.roster import OcrOnlySource, RosterSource
    from nfl_gsplat.paths import play_paths
    from nfl_gsplat.utils.plays import load_plays

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(game: str = typer.Option(...), play: str = typer.Option(...),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pp = play_paths(cfg, game, play)
        manifest = load_plays(pp.game.plays_yaml)
        home, away = manifest.game_teams
        df = pd_.read_parquet(pp.tracks)
        video_paths = {cam: pp.game.raw_video(cam) for cam in pp.game.cameras}

        if str(cfg.identity.source) == "roster":
            source = RosterSource.from_parquet(
                str(cfg.identity.season), pp.game.rosters_dir,
                game_teams={game: (home, away)})
        else:
            source = OcrOnlySource()
        candidates = source.candidates_for_play(game, play)

        match_cfg = IdentityMatchConfig(
            season=str(cfg.identity.season),
            jersey_weight=float(cfg.identity.match.jersey_weight),
            team_mismatch_cost=float(cfg.identity.match.team_mismatch_cost),
            unknown_jersey_cost=float(cfg.identity.match.unknown_jersey_cost),
            max_match_cost=float(cfg.identity.match.max_match_cost),
        )
        ref_cfg = (RefereeConfig(
            min_stripe_transitions=int(cfg.identity.referee.min_stripe_transitions),
            max_mean_saturation=float(cfg.identity.referee.max_mean_saturation))
            if bool(cfg.identity.referee.enabled) else None)

        crops = extract_representative_crops(df, video_paths, id_col="global_player_id")
        _, assignments = assign_play_identities(
            df, crops, candidates, home, away, match_cfg,
            id_col="global_player_id", ref_cfg=ref_cfg)
        write_entities_json(pp.entities, assignments)

    app()


if __name__ == "__main__":
    _main()
