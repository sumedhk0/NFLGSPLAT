# Per-Play Season Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-game continuous-video + `plays.yaml` frame-window layout with a self-contained `data/{season}/week_NN/AWAY_at_HOME/play_NNN/` tree where each play folder holds its own clips, per-play calibration + field, and all derived data.

**Architecture:** A single `PlayDir` resolver (paths.py) keys every artifact off one play directory, with three `_`-prefixed season-shared roots for the cross-play caches. A `meta.yaml` per play (replacing `plays.yaml`) carries fps/teams/gsis. Stages read the whole clip (no frame windows) and take a single `--play-dir` arg. The season DAG discovers plays by walking the tree.

**Tech Stack:** Python 3.10, dataclasses, OmegaConf (YAML), typer (CLIs), pytest. CPU-only changes; ruff-clean; follow existing `nfl_gsplat` patterns (fail-loud `SetupError`, `utils.io`, `utils.logging`).

**Reference spec:** `docs/superpowers/specs/2026-06-12-per-play-season-tree-design.md`

---

## Conventions used throughout

- Run all commands from the repo root: `C:/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/NFLGSPLAT`.
- Run tests with the local interpreter: `python -m pytest <args>` (on PACE use `conda run -n nfl_smplx python -m pytest ...`).
- `matchup` string is always `"{away}_at_{home}"` (e.g. `NO_at_ATL`). `PlayDir.teams` returns `(home, away)`.
- Commit messages end with the project trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Phase 1 — Path resolver + meta loader

### Task 1: `PlayDir` resolver in `paths.py`

**Files:**
- Modify: `nfl_gsplat/paths.py` (replace `GamePaths`/`PlayPaths`/`game_paths`/`play_paths`)
- Test: `tests/test_config_paths.py` (replace the two layout tests)

- [ ] **Step 1: Write the failing tests** — replace `test_game_paths_layout` and `test_play_paths_layout` in `tests/test_config_paths.py` with:

```python
def test_play_dir_layout():
    from nfl_gsplat.paths import PlayDir
    pd = PlayDir(season="2024", week=1, matchup="NO_at_ATL", play_id="play_001")
    assert pd.dir == Path("data/2024/week_01/NO_at_ATL/play_001")
    assert pd.video("sideline") == Path("data/2024/week_01/NO_at_ATL/play_001/sideline.mp4")
    assert pd.cameras_json == Path("data/2024/week_01/NO_at_ATL/play_001/cameras.json")
    assert pd.field_ply == Path("data/2024/week_01/NO_at_ATL/play_001/field.ply")
    assert pd.tracks == Path("data/2024/week_01/NO_at_ATL/play_001/tracks.parquet")
    assert pd.entities == Path("data/2024/week_01/NO_at_ATL/play_001/entities.json")
    assert pd.pose("00-1234") == Path("data/2024/week_01/NO_at_ATL/play_001/poses/00-1234.npz")
    assert pd.ball == Path("data/2024/week_01/NO_at_ATL/play_001/ball.npz")
    assert pd.render_mp4 == Path("data/2024/week_01/NO_at_ATL/play_001/render.mp4")
    assert pd.meta_yaml == Path("data/2024/week_01/NO_at_ATL/play_001/meta.yaml")


def test_play_dir_season_shared_roots():
    from nfl_gsplat.paths import PlayDir
    pd = PlayDir(season="2024", week=12, matchup="NO_at_ATL", play_id="play_003")
    assert pd.library_root == Path("data/2024/_library")
    assert pd.rosters_root == Path("data/2024/_rosters")
    assert pd.registry_path == Path("data/2024/_registry.json")
    assert pd.teams == ("ATL", "NO")          # (home, away)


def test_play_dir_from_dir_roundtrip(tmp_path):
    from nfl_gsplat.paths import PlayDir
    p = tmp_path / "data" / "2024" / "week_05" / "NO_at_ATL" / "play_002"
    p.mkdir(parents=True)
    pd = PlayDir.from_dir(p)
    assert (pd.season, pd.week, pd.matchup, pd.play_id) == ("2024", 5, "NO_at_ATL", "play_002")
    assert pd.dir == p
    assert pd.library_root == tmp_path / "data" / "2024" / "_library"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config_paths.py -q`
Expected: FAIL with `ImportError: cannot import name 'PlayDir'`.

- [ ] **Step 3: Rewrite `nfl_gsplat/paths.py`** — replace the entire file body (keep the module docstring style) with:

```python
"""Single source of truth for every on-disk artifact path.

Stages never hand-build paths; they construct a :class:`PlayDir` (from an
explicit play directory or from config) and read its attributes. Every play is
self-contained under ``data/{season}/week_NN/{away}_at_{home}/play_NNN/``; the
only cross-play state lives in three season-shared roots prefixed ``_``.

Layout::

    data/{season}/week_NN/{matchup}/play_NNN/sideline.mp4   video(cam)
    data/{season}/week_NN/{matchup}/play_NNN/endzone.mp4
    data/{season}/week_NN/{matchup}/play_NNN/cameras.json   cameras_json (per-play calib)
    data/{season}/week_NN/{matchup}/play_NNN/field.ply      field_ply    (per-play field)
    data/{season}/week_NN/{matchup}/play_NNN/tracks.parquet tracks
    data/{season}/week_NN/{matchup}/play_NNN/entities.json  entities
    data/{season}/week_NN/{matchup}/play_NNN/smplestx/      smplestx_dir
    data/{season}/week_NN/{matchup}/play_NNN/poses/{uid}.npz pose(uid)
    data/{season}/week_NN/{matchup}/play_NNN/ball.npz       ball
    data/{season}/week_NN/{matchup}/play_NNN/render.mp4     render_mp4
    data/{season}/week_NN/{matchup}/play_NNN/meta.yaml      meta_yaml
    data/{season}/_library/                                 library_root  (cross-play)
    data/{season}/_rosters/                                 rosters_root
    data/{season}/_registry.json                            registry_path

``matchup`` is ``"{away}_at_{home}"`` (NFL-standard). :attr:`PlayDir.teams`
returns ``(home, away)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _select(cfg: DictConfig, key: str, default):
    val = OmegaConf.select(cfg, key)
    return default if val is None else val


@dataclass(frozen=True)
class PlayDir:
    """Resolver for every artifact of one play, plus the season-shared roots."""

    season: str
    week: int
    matchup: str                 # "{away}_at_{home}"
    play_id: str                 # "play_001"
    data_root: Path = Path("data")
    cameras: tuple[str, ...] = ("sideline", "endzone")

    # --- the play folder + its artifacts -----------------------------------

    @property
    def season_root(self) -> Path:
        return self.data_root / str(self.season)

    @property
    def week_dir(self) -> Path:
        return self.season_root / f"week_{int(self.week):02d}"

    @property
    def matchup_dir(self) -> Path:
        return self.week_dir / self.matchup

    @property
    def dir(self) -> Path:
        return self.matchup_dir / self.play_id

    def video(self, cam: str) -> Path:
        return self.dir / f"{cam}.mp4"

    @property
    def cameras_json(self) -> Path:
        return self.dir / "cameras.json"

    @property
    def field_ply(self) -> Path:
        return self.dir / "field.ply"

    @property
    def tracks(self) -> Path:
        return self.dir / "tracks.parquet"

    @property
    def entities(self) -> Path:
        return self.dir / "entities.json"

    @property
    def smplestx_dir(self) -> Path:
        return self.dir / "smplestx"

    @property
    def joints3d(self) -> Path:
        return self.dir / "joints3d.npz"

    @property
    def poses_dir(self) -> Path:
        return self.dir / "poses"

    def pose(self, uid: str) -> Path:
        return self.poses_dir / f"{uid}.npz"

    @property
    def ball(self) -> Path:
        return self.dir / "ball.npz"

    @property
    def render_mp4(self) -> Path:
        return self.dir / "render.mp4"

    @property
    def meta_yaml(self) -> Path:
        return self.dir / "meta.yaml"

    # --- season-shared (cross-play) roots ----------------------------------

    @property
    def library_root(self) -> Path:
        return self.season_root / "_library"

    @property
    def rosters_root(self) -> Path:
        return self.season_root / "_rosters"

    @property
    def registry_path(self) -> Path:
        return self.season_root / "_registry.json"

    # --- derived metadata ---------------------------------------------------

    @property
    def teams(self) -> tuple[str, str]:
        """``(home, away)`` parsed from the matchup ``{away}_at_{home}``."""
        away, home = self.matchup.split("_at_")
        return home, away

    # --- constructors -------------------------------------------------------

    @classmethod
    def from_dir(cls, path, *, cameras: tuple[str, ...] = ("sideline", "endzone")) -> "PlayDir":
        """Build a :class:`PlayDir` from an existing play directory path.

        Expects ``.../{data_root}/{season}/week_NN/{matchup}/play_NNN``.
        """
        p = Path(path)
        play_id = p.name
        matchup = p.parent.name
        week_name = p.parent.parent.name
        if not week_name.startswith("week_"):
            raise ValueError(f"{path}: expected a week_NN folder, got {week_name!r}")
        week = int(week_name[len("week_"):])
        season = p.parent.parent.parent.name
        data_root = p.parent.parent.parent.parent
        return cls(season=season, week=week, matchup=matchup, play_id=play_id,
                   data_root=data_root, cameras=tuple(cameras))


def play_dir(cfg: DictConfig, season, week: int, matchup: str, play_id: str) -> PlayDir:
    """Construct a :class:`PlayDir` from config defaults (data root + cameras)."""
    cams = tuple(str(c) for c in _select(cfg, "cameras", ["sideline", "endzone"]))
    return PlayDir(
        season=str(season),
        week=int(week),
        matchup=str(matchup),
        play_id=str(play_id),
        data_root=Path(str(_select(cfg, "paths.data_root", "data"))),
        cameras=cams,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_paths.py -q`
Expected: PASS (the three new tests plus the unchanged config tests).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/paths.py tests/test_config_paths.py
git add nfl_gsplat/paths.py tests/test_config_paths.py
git commit -m "Replace GamePaths/PlayPaths with self-contained PlayDir resolver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `meta.py` loader (replaces `plays.py`)

**Files:**
- Create: `nfl_gsplat/utils/meta.py`
- Delete: `nfl_gsplat/utils/plays.py` (after consumers migrate — see Phase 3; deletion happens in Task 11)
- Test: `tests/test_meta.py` (new), delete `tests/test_plays.py` in Task 11

- [ ] **Step 1: Write the failing test** — create `tests/test_meta.py`:

```python
from __future__ import annotations

import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.meta import PlayMeta, load_meta

_VALID = """\
season: 2024
week: 1
home_team: ATL
away_team: "NO"
fps: 30.0
gsis_play_id: 36
"""


def test_load_meta_valid(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text(_VALID)
    m = load_meta(p)
    assert isinstance(m, PlayMeta)
    assert m.season == "2024"
    assert m.week == 1
    assert m.game_teams == ("ATL", "NO")
    assert m.fps == 30.0
    assert m.gsis_play_id == "36"


def test_load_meta_missing_file(tmp_path):
    with pytest.raises(SetupError, match="meta.yaml"):
        load_meta(tmp_path / "nope.yaml")


def test_load_meta_boolean_team_abbrev(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text("season: 2024\nweek: 1\nhome_team: ATL\naway_team: NO\nfps: 30\n")
    with pytest.raises(SetupError, match="quote"):
        load_meta(p)


def test_load_meta_missing_required(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text("season: 2024\nweek: 1\nhome_team: ATL\nfps: 30\n")
    with pytest.raises(SetupError, match="away_team"):
        load_meta(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_meta.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_gsplat.utils.meta'`.

- [ ] **Step 3: Create `nfl_gsplat/utils/meta.py`**:

```python
"""Per-play metadata (``meta.yaml``) — fps + teams + optional gsis play id.

One ``meta.yaml`` lives in each play folder
(``data/{season}/week_NN/{matchup}/play_NNN/meta.yaml``). Season/week/teams are
also encoded in the path, but this file is the authoritative record and carries
``fps`` and ``gsis_play_id``, which the path does not. Replaces the old
``plays.yaml`` frame-window manifest (plays are now standalone clips).

Schema::

    season: 2024
    week: 1
    home_team: ATL
    away_team: "NO"      # quote abbreviations: bare NO/ON/NA parse as booleans
    fps: 30.0
    gsis_play_id: 36     # optional; nflverse participation alignment only
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from nfl_gsplat.errors import SetupError


@dataclass(frozen=True)
class PlayMeta:
    season: str
    week: int
    home_team: str
    away_team: str
    fps: float
    gsis_play_id: str | None = None

    @property
    def game_teams(self) -> tuple[str, str]:
        return (self.home_team, self.away_team)


def load_meta(path) -> PlayMeta:
    """Load + validate a play's ``meta.yaml`` (fail-loud per project philosophy)."""
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"play meta.yaml missing at {path}. Create it (season/week/home_team/"
            "away_team/fps) — see SETUP.md §5. Use scripts/new_play.py to scaffold one."
        )
    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(raw, dict):
        raise SetupError(f"{path}: expected a mapping of meta fields.")
    for key in ("season", "week", "home_team", "away_team"):
        if key not in raw:
            raise SetupError(f"{path}: meta.{key} is required.")
    # YAML 1.1 coerces NO / NA / ON / yes / off to booleans — a footgun for team
    # abbreviations like "NO" (New Orleans). Fail loud and tell the user to quote.
    for key in ("home_team", "away_team"):
        if isinstance(raw[key], bool):
            raise SetupError(
                f"{path}: meta.{key} parsed as a boolean — quote the abbreviation "
                f'(e.g. {key}: "NO") so YAML keeps it a string.'
            )
    gsis = raw.get("gsis_play_id")
    return PlayMeta(
        season=str(raw["season"]),
        week=int(raw["week"]),
        home_team=str(raw["home_team"]),
        away_team=str(raw["away_team"]),
        fps=float(raw.get("fps", 30.0)),
        gsis_play_id=str(gsis) if gsis is not None else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_meta.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/utils/meta.py tests/test_meta.py
git add nfl_gsplat/utils/meta.py tests/test_meta.py
git commit -m "Add per-play meta.yaml loader (replaces plays.yaml frame windows)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 — Season discovery + scaffolder

### Task 3: Tree-walk play discovery

**Files:**
- Create: `nfl_gsplat/season/discover.py`
- Test: `tests/test_discover.py` (new)

- [ ] **Step 1: Write the failing test** — create `tests/test_discover.py`:

```python
from __future__ import annotations

from nfl_gsplat.season.discover import discover_plays


def _make_play(root, season, week, matchup, play, *, full=True):
    d = root / "data" / season / f"week_{week:02d}" / matchup / play
    d.mkdir(parents=True)
    if full:
        (d / "sideline.mp4").write_bytes(b"x")
        (d / "endzone.mp4").write_bytes(b"x")
        (d / "meta.yaml").write_text("season: 2024\nweek: 1\nhome_team: ATL\naway_team: \"NO\"\nfps: 30\n")
    return d


def test_discover_orders_week_matchup_play(tmp_path):
    _make_play(tmp_path, "2024", 2, "NO_at_ATL", "play_001")
    _make_play(tmp_path, "2024", 1, "GB_at_CHI", "play_002")
    _make_play(tmp_path, "2024", 1, "GB_at_CHI", "play_001")
    plays = discover_plays(tmp_path / "data", "2024")
    got = [(p.week, p.matchup, p.play_id) for p in plays]
    assert got == [
        (1, "GB_at_CHI", "play_001"),
        (1, "GB_at_CHI", "play_002"),
        (2, "NO_at_ATL", "play_001"),
    ]


def test_discover_skips_incomplete(tmp_path, caplog):
    _make_play(tmp_path, "2024", 1, "NO_at_ATL", "play_001")          # complete
    _make_play(tmp_path, "2024", 1, "NO_at_ATL", "play_002", full=False)  # no videos/meta
    plays = discover_plays(tmp_path / "data", "2024")
    assert [p.play_id for p in plays] == ["play_001"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discover.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_gsplat.season.discover'`.

- [ ] **Step 3: Create `nfl_gsplat/season/discover.py`**:

```python
"""Discover the plays of a season by walking the on-disk tree.

The filesystem is the source of truth: every directory matching
``data/{season}/week_NN/{away}_at_{home}/play_NNN`` that contains both clips and
a ``meta.yaml`` is a play. Drives the season DAG (no explicit games manifest).
"""
from __future__ import annotations

from pathlib import Path

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

_REQUIRED = ("sideline.mp4", "endzone.mp4", "meta.yaml")


def discover_plays(data_root, season, *, cameras=("sideline", "endzone")) -> list[PlayDir]:
    """Return the season's plays as :class:`PlayDir`s, ordered week→matchup→play.

    Directories missing any required file (both clips + meta.yaml) are skipped
    with a warning so a half-uploaded play never silently enters the DAG.
    """
    root = Path(data_root) / str(season)
    plays: list[PlayDir] = []
    for play in sorted(root.glob("week_*/*_at_*/play_*")):
        if not play.is_dir():
            continue
        missing = [f for f in _REQUIRED if not (play / f).exists()]
        if missing:
            _LOG.warning(f"discover: skipping {play} (missing {missing})")
            continue
        plays.append(PlayDir.from_dir(play, cameras=tuple(cameras)))
    return plays
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discover.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/season/discover.py tests/test_discover.py
git add nfl_gsplat/season/discover.py tests/test_discover.py
git commit -m "Add tree-walk season play discovery

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `new_play.py` scaffolder

**Files:**
- Create: `scripts/new_play.py`
- Test: `tests/test_new_play.py` (new)

- [ ] **Step 1: Write the failing test** — create `tests/test_new_play.py`:

```python
from __future__ import annotations

import pytest

from nfl_gsplat.season.scaffold import scaffold_play
from nfl_gsplat.utils.meta import load_meta


def test_scaffold_creates_folder_and_meta(tmp_path):
    pd = scaffold_play(tmp_path / "data", season="2024", week=1,
                       away="NO", home="ATL", play="play_001", fps=30.0,
                       gsis_play_id="36")
    assert pd.dir == tmp_path / "data" / "2024" / "week_01" / "NO_at_ATL" / "play_001"
    assert pd.dir.is_dir()
    m = load_meta(pd.meta_yaml)
    assert m.game_teams == ("ATL", "NO")
    assert m.fps == 30.0
    assert m.gsis_play_id == "36"


def test_scaffold_refuses_overwrite(tmp_path):
    scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                  home="ATL", play="play_001")
    with pytest.raises(FileExistsError):
        scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                      home="ATL", play="play_001")


def test_scaffold_force_overwrites(tmp_path):
    scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                  home="ATL", play="play_001", fps=30.0)
    pd = scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                       home="ATL", play="play_001", fps=60.0, force=True)
    assert load_meta(pd.meta_yaml).fps == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_new_play.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_gsplat.season.scaffold'`.

- [ ] **Step 3: Create the core `nfl_gsplat/season/scaffold.py`** (testable core, used by the CLI):

```python
"""Scaffold a new play folder + meta.yaml stub. Core for scripts/new_play.py."""
from __future__ import annotations

from pathlib import Path

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def scaffold_play(
    data_root,
    *,
    season,
    week: int,
    away: str,
    home: str,
    play: str,
    fps: float = 30.0,
    gsis_play_id: str | None = None,
    force: bool = False,
) -> PlayDir:
    """Create ``data/{season}/week_NN/{away}_at_{home}/{play}/`` + a meta.yaml stub.

    Raises :class:`FileExistsError` if meta.yaml already exists and ``force`` is
    False. Returns the resolved :class:`PlayDir`. The user drops the two clips
    into ``pd.dir`` afterward.
    """
    pd = PlayDir(season=str(season), week=int(week), matchup=f"{away}_at_{home}",
                 play_id=str(play), data_root=Path(data_root))
    pd.dir.mkdir(parents=True, exist_ok=True)
    if pd.meta_yaml.exists() and not force:
        raise FileExistsError(
            f"{pd.meta_yaml} already exists; pass force=True to overwrite."
        )
    lines = [
        f"season: {season}",
        f"week: {int(week)}",
        f"home_team: {home}",
        f'away_team: "{away}"',
        f"fps: {fps}",
    ]
    if gsis_play_id is not None:
        lines.append(f"gsis_play_id: {gsis_play_id}")
    pd.meta_yaml.write_text("\n".join(lines) + "\n")
    _LOG.info(f"scaffolded play → {pd.dir} (drop sideline.mp4 + endzone.mp4 here)")
    return pd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_new_play.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Create the CLI `scripts/new_play.py`** (thin wrapper; not unit-tested):

```python
"""Scaffold a new play folder + meta.yaml stub.

    python scripts/new_play.py --season 2024 --week 1 --away NO --home ATL \
        --play play_001 --fps 30 [--gsis-play-id 36] [--force]

Creates data/{season}/week_NN/{away}_at_{home}/{play}/ and a meta.yaml stub;
drop sideline.mp4 + endzone.mp4 into the printed folder afterward.
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.season.scaffold import scaffold_play

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    season: str = typer.Option(...),
    week: int = typer.Option(...),
    away: str = typer.Option(...),
    home: str = typer.Option(...),
    play: str = typer.Option("play_001"),
    fps: float = typer.Option(30.0),
    gsis_play_id: str = typer.Option(None, "--gsis-play-id"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    pd = scaffold_play(data_root, season=season, week=week, away=away, home=home,
                       play=play, fps=fps, gsis_play_id=gsis_play_id, force=force)
    print(f"created {pd.dir}")
    print(f"  -> drop {pd.video('sideline')} and {pd.video('endzone')}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Lint + commit**

```bash
python -m ruff check nfl_gsplat/season/scaffold.py scripts/new_play.py tests/test_new_play.py
git add nfl_gsplat/season/scaffold.py scripts/new_play.py tests/test_new_play.py
git commit -m "Add new_play.py scaffolder (play folder + meta.yaml stub)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — Migrate stage CLIs to `--play-dir`

Every stage CLI currently takes `--game --play --config`, builds
`play_paths(cfg, game, play)`, and (most) call `load_plays(pp.game.plays_yaml)`.
The migration is identical in shape for each: take a single `--play-dir PATH`,
build `pd = PlayDir.from_dir(play_dir)`, read the whole clip (no window), and use
`pd.*` artifact paths. `detect_track` additionally drops its frame-window slice.

> Each task below shows the exact replacement for that file's `_main()`. Apply
> only the `_main()` (CLI) edits — the pure cores (`window_tracks`,
> `solve_joint_tfms`, etc.) are unchanged.

### Task 5: `detect_track._main`

**Files:**
- Modify: `nfl_gsplat/tracking/detect_track.py` (`_main`, lines ~135-162)

- [ ] **Step 1: Replace the `_main` body** with:

```python
def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    from pathlib import Path

    import typer

    from nfl_gsplat.config import load_config
    from nfl_gsplat.paths import PlayDir

    app = typer.Typer(add_completion=False)

    @app.command()
    def run(play_dir: Path = typer.Option(..., "--play-dir"),
            config: Path = typer.Option(Path("configs/pipeline.yaml"))):
        cfg = load_config(config)
        pd = PlayDir.from_dir(play_dir)
        tcfg = _track_cfg_from(cfg)  # unchanged helper if present; else inline as before
        dfs = [detect_and_track(pd.video(cam), cam, tcfg) for cam in pd.cameras]
        df = _concat_tracks(dfs)  # whole clip — no window slice
        pd.dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pd.tracks, index=False)
        _LOG.info(f"detect_track: {len(df)} detections → {pd.tracks}")

    app()
```

> NOTE: keep whatever local helpers the current `_main` used to build `tcfg` and
> concatenate `dfs`; the only behavioral change is dropping
> `load_plays(...).window(play)` and the `[start,end]` slice — the stage now
> processes the full clip. If the current code concatenated inline, keep that
> inline; do not invent `_track_cfg_from`/`_concat_tracks` if they don't already
> exist — reuse the exact expressions from the original `_main`.

- [ ] **Step 2: Verify import + CLI wiring compiles**

Run: `python -c "import nfl_gsplat.tracking.detect_track"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nfl_gsplat/tracking/detect_track.py
git commit -m "detect_track CLI: --play-dir, whole-clip (drop frame window)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 6: `cross_cam_reid._main` and `jersey_ocr._main`

**Files:**
- Modify: `nfl_gsplat/tracking/cross_cam_reid.py` (`_main`)
- Modify: `nfl_gsplat/tracking/jersey_ocr.py` (`_main`)

- [ ] **Step 1: For each file, change the CLI signature + path build** to the
  same pattern: `--play-dir PATH` → `pd = PlayDir.from_dir(play_dir)`, replace
  `pp.game.raw_video(cam)`→`pd.video(cam)`, `pp.tracks`→`pd.tracks`, drop any
  `load_plays`/window use. Imports become:

```python
    from nfl_gsplat.config import load_config
    from nfl_gsplat.paths import PlayDir
```

- [ ] **Step 2: Verify both import**

Run: `python -c "import nfl_gsplat.tracking.cross_cam_reid, nfl_gsplat.tracking.jersey_ocr"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nfl_gsplat/tracking/cross_cam_reid.py nfl_gsplat/tracking/jersey_ocr.py
git commit -m "reid + jersey_ocr CLIs: --play-dir, whole-clip

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 7: `assign_stage._main`

**Files:**
- Modify: `nfl_gsplat/identity/assign_stage.py` (`_main`, lines ~202-247)

- [ ] **Step 1: Replace path/meta wiring.** Build `pd = PlayDir.from_dir(play_dir)`
  and `meta = load_meta(pd.meta_yaml)`; use `meta.game_teams` where the code
  currently read `manifest.game_teams`, `pd.video(cam)` for `pp.game.raw_video`,
  `pd.tracks`/`pd.entities` for the artifacts. Imports:

```python
    from nfl_gsplat.config import load_config
    from nfl_gsplat.paths import PlayDir
    from nfl_gsplat.utils.meta import load_meta
```

- [ ] **Step 2: Verify import**

Run: `python -c "import nfl_gsplat.identity.assign_stage"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nfl_gsplat/identity/assign_stage.py
git commit -m "assign_stage CLI: --play-dir + meta.yaml teams

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 8: `run_pose._main` and `run_ball._main`

**Files:**
- Modify: `nfl_gsplat/pose/run_pose.py` (`_main`, lines ~195-224+)
- Modify: `nfl_gsplat/ball/run_ball.py` (`_main`, lines ~43-72+)

- [ ] **Step 1:** In both, replace `pp = play_paths(cfg, game, play)` +
  `load_plays(pp.game.plays_yaml)` with `pd = PlayDir.from_dir(play_dir)` +
  (where fps/teams are needed) `meta = load_meta(pd.meta_yaml)`. Map
  `pp.game.raw_video(cam)`→`pd.video(cam)`, `pp.tracks`→`pd.tracks`,
  `pp.entities`→`pd.entities`, `pp.dir`→`pd.dir`, and any
  `pp.poses_dir/pp.pose(uid)`→`pd.poses_dir/pd.pose(uid)`,
  `library` root → `pd.library_root`. Imports as in Task 7.

- [ ] **Step 2: Verify both import**

Run: `python -c "import nfl_gsplat.pose.run_pose, nfl_gsplat.ball.run_ball"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nfl_gsplat/pose/run_pose.py nfl_gsplat/ball/run_ball.py
git commit -m "run_pose + run_ball CLIs: --play-dir + season-shared library root

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 9: `build_play._main`

**Files:**
- Modify: `nfl_gsplat/avatars/build_play.py` (`_main`, lines ~66-81+)

- [ ] **Step 1:** Replace `pp = play_paths(cfg, game, play)` with
  `pd = PlayDir.from_dir(play_dir)`; `pp.entities`→`pd.entities`; build the
  `AvatarLibrary` rooted at `pd.library_root` with `season=""` (the season dir is
  already in the path — see the `_library` reconciliation note below). Imports as
  in Task 7.

- [ ] **Step 2: Verify import**

Run: `python -c "import nfl_gsplat.avatars.build_play"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nfl_gsplat/avatars/build_play.py
git commit -m "build_play CLI: --play-dir + season-shared library root

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> **`_library` reconciliation (applies to Tasks 8-9 and Task 12):**
> `AvatarLibrary(root, season)` builds `{root}/{season}/{uid}/...`. With the new
> layout the season is already in `pd.library_root` (`data/{season}/_library`), so
> construct it as `AvatarLibrary(root=pd.library_root, season="")` to avoid
> `_library/2024/...`. The `_player_dir` for `season=""` yields
> `_library/<uid>` — verify in Task 12's test.

---

## Phase 4 — DAG + season driver + calibration

### Task 10: Per-play DAG (fold field into perception, discover plays)

**Files:**
- Modify: `nfl_gsplat/season/dag.py` (`build_submission_plan`, `num_plays_range`)
- Modify: `scripts/run_season.py`
- Modify: `configs/season.yaml` (drop `games:`; add `paths.data_root`)
- Test: `tests/test_season_dag.py` (update assertions)

- [ ] **Step 1: Update the DAG test** — replace the games-based plan assertions in
  `tests/test_season_dag.py` with a discovery-based plan. The new
  `build_submission_plan(cfg, plays)` takes the discovered `list[PlayDir]`:

```python
def test_plan_is_per_play_with_field_folded(tmp_cfg, three_plays):
    from nfl_gsplat.season.dag import build_submission_plan
    plan = build_submission_plan(tmp_cfg, three_plays)
    text = "\n".join(plan)
    # No separate field stage; perception runs per play and builds its own field.
    assert "field_recon.sbatch" not in text
    assert "perception_array.sbatch" in text
    assert text.count("--qos=embers") >= 1
    assert "collect_uids" in text
    assert "avatar_build_array.sbatch" in text
    assert "render_array.sbatch" in text
```

(Keep the existing `_cfg()` fixture; add a `three_plays` fixture returning three
`PlayDir`s via `PlayDir(season="2024", week=1, matchup="NO_at_ATL", play_id=f"play_00{i}")`.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_season_dag.py -q`
Expected: FAIL (signature mismatch / field assertions).

- [ ] **Step 3: Rewrite `build_submission_plan`** in `nfl_gsplat/season/dag.py` to
  take discovered plays and emit a per-play perception array (no field stage):

```python
def build_submission_plan(cfg, plays: list) -> list[str]:
    """Ordered sbatch strings for the per-play season DAG.

    ``plays`` is the discovered ``list[PlayDir]``. Perception runs once per play
    (building that play's own field + calibration-driven reconstruction), then a
    CPU tail collects uids and submits the avatar-build array, then render runs
    per play.
    """
    season = str(cfg.season)
    flags = " ".join(slurm_flags(cfg))
    qos = " ".join(qos_flags(cfg))
    qos = f"{qos} " if qos else ""
    n = max(len(plays), 1)
    plan: list[str] = []
    # S2 perception — one array task per play; PlayDir paths passed via env list.
    plan.append(
        f"sbatch --parsable {flags} --time={cfg.slurm.time_perception} "
        f"--array=1-{n} --export=ALL,NFL_SEASON={season} "
        f"scripts/slurm/perception_array.sbatch   # S2 perception[{n} plays]"
    )
    plan.append(
        f"sbatch --parsable {qos}--dependency=afterok:$PERCEPTION "
        f'--wrap="python -m nfl_gsplat.season.collect_uids --season {season} && '
        f"sbatch {flags} --time={cfg.slurm.time_avatar} "
        "--array=1-$(wc -l < outputs/avatar_worklist.txt) "
        f'--export=ALL,NFL_SEASON={season} scripts/slurm/avatar_build_array.sbatch"   # tail: collect + S3'
    )
    plan.append(
        f"sbatch {flags} --time={cfg.slurm.time_render} "
        f"--dependency=afterok:$AVATAR_JOB --array=1-{n} "
        f"--export=ALL,NFL_SEASON={season} "
        f"scripts/slurm/render_array.sbatch   # S4 render[{n} plays]"
    )
    return plan
```

Delete `num_plays_range` (no longer used).

- [ ] **Step 4: Update `scripts/run_season.py`** to discover plays and write the
  per-play worklist the array jobs read:

```python
@app.command()
def main(
    config: Path = typer.Option(Path("configs/season.yaml")),
    dry_run: bool = typer.Option(True, "--dry-run/--submit"),
) -> None:
    cfg = OmegaConf.load(str(config))
    from nfl_gsplat.season.discover import discover_plays
    data_root = OmegaConf.select(cfg, "paths.data_root") or "data"
    plays = discover_plays(data_root, cfg.season)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/play_worklist.txt").write_text(
        "\n".join(str(p.dir) for p in plays) + ("\n" if plays else "")
    )
    plan = build_submission_plan(cfg, plays)
    print(f"# Season DAG plan ({len(plays)} plays):\n")
    for step in plan:
        print(step)
    if dry_run:
        print("\n# dry run — nothing submitted. Re-run with --submit.")
        return
    raise typer.Exit(
        code=subprocess.call(["bash", "-c", "echo 'submit path: wire job-id capture on PACE'"])
    )
```

- [ ] **Step 5: Edit `configs/season.yaml`** — remove the `games:` block; add
  `paths.data_root`:

```yaml
season: 2024

paths:
  data_root: data        # season tree root: data/{season}/week_NN/{matchup}/play_NNN

slurm:
  account: gatech
  partition: gpu-h100
  gpu: "h100:1"
  qos: embers
  requeue: true
  cpus_per_task: 8
  mem: "64G"
  time_perception: "01:00:00"
  time_avatar: "04:00:00"
  time_render: "01:00:00"
```

- [ ] **Step 6: Run the DAG test**

Run: `python -m pytest tests/test_season_dag.py -q`
Expected: PASS.

- [ ] **Step 7: Lint + commit**

```bash
python -m ruff check nfl_gsplat/season/dag.py scripts/run_season.py tests/test_season_dag.py
git add nfl_gsplat/season/dag.py scripts/run_season.py configs/season.yaml tests/test_season_dag.py
git commit -m "Per-play season DAG: discover plays, fold field into perception

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 11: Retire `plays.py`; retarget calibration; update alignment + tier2 test

**Files:**
- Delete: `nfl_gsplat/utils/plays.py`, `tests/test_plays.py`
- Modify: `nfl_gsplat/identity/alignment.py` (swap `load_plays`→`load_meta`/PlayDir)
- Modify: `scripts/02_calibrate_cameras.py` (`--game`→`--play-dir`, write `pd.cameras_json`)
- Modify: `tests/test_season_tier2.py` (drop `play_paths`/`load_plays` usage)

- [ ] **Step 1: Migrate `alignment.py`** — it currently uses `load_plays` for the
  `gsis_play_id` → participation map. Switch to `load_meta(pd.meta_yaml)`:
  read `meta.gsis_play_id` per play instead of iterating a manifest's windows.
  Verify any `PlayWindow`/`PlaysManifest` references are removed.

- [ ] **Step 2: Retarget `scripts/02_calibrate_cameras.py`** — replace
  `--game GAME` with `--play-dir PATH`; build `pd = PlayDir.from_dir(play_dir)`;
  read frames from `pd.video(cam)`; write the result to `pd.cameras_json` (was
  `gp.calib_json`).

- [ ] **Step 3: Update `tests/test_season_tier2.py`** — replace any
  `play_paths(...)`/`load_plays(...)` with `PlayDir(...)` and a written
  `meta.yaml` via the `scaffold_play` helper from Task 4 (import
  `from nfl_gsplat.season.scaffold import scaffold_play`). Keep the tier-2
  behavior assertions; only the path/meta plumbing changes.

- [ ] **Step 4: Delete the retired modules**

```bash
git rm nfl_gsplat/utils/plays.py tests/test_plays.py
```

- [ ] **Step 5: Grep for stragglers**

Run (PowerShell-safe via the Grep tool or):
`python -c "import nfl_gsplat.identity.alignment"`
Then search the tree for any remaining `plays_yaml|load_plays|PlayWindow|game_paths|play_paths|PlaysManifest` — there must be **zero** matches outside the spec/plan docs.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Retire plays.py; retarget calibration + alignment to PlayDir/meta

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 5 — Library reconciliation + full green + docs

### Task 12: `_library` season-subdir reconciliation

**Files:**
- Modify: `nfl_gsplat/avatars/library.py` (handle `season=""` cleanly)
- Test: `tests/test_avatar_library.py` (add a `season=""` layout test)

- [ ] **Step 1: Write the failing test** — add to `tests/test_avatar_library.py`:

```python
def test_library_empty_season_flat_layout(tmp_path):
    from nfl_gsplat.avatars.library import AvatarLibrary
    from nfl_gsplat.avatars.lhm_wrapper import write_mock_avatar

    lib = AvatarLibrary(root=tmp_path / "_library", season="")
    av = {}  # build a minimal canonical avatar via the mock writer
    out = write_mock_avatar(tmp_path / "mock.npz", num_gaussians=64, num_joints=22)
    import numpy as np
    from nfl_gsplat.utils.io import read_npz
    lib.put_avatar("p_7", read_npz(out))
    assert (tmp_path / "_library" / "p_7" / "avatar.npz").exists()
    assert lib.has_avatar("p_7")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_avatar_library.py::test_library_empty_season_flat_layout -q`
Expected: FAIL (path has an extra empty-string segment or the file isn't where asserted).

- [ ] **Step 3: Fix `_player_dir` in `library.py`** to skip an empty season
  segment:

```python
    def _player_dir(self, uid: str) -> Path:
        if uid == REFEREE_UID:
            base = self.root / self.season if self.season else self.root
            return base / "_assets" / "referee"
        if uid == FOOTBALL_UID:
            return self.root / "_assets" / "football"
        return (self.root / self.season / uid) if self.season else (self.root / uid)
```

- [ ] **Step 4: Run the library tests**

Run: `python -m pytest tests/test_avatar_library.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
python -m ruff check nfl_gsplat/avatars/library.py tests/test_avatar_library.py
git add nfl_gsplat/avatars/library.py tests/test_avatar_library.py
git commit -m "AvatarLibrary: flat layout when season is empty (data/{season}/_library)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 13: Full suite green + SETUP.md docs

**Files:**
- Modify: `SETUP.md` (§5 plays.yaml → meta.yaml + tree; §9 DAG; §10 stage CLI table `--play-dir`)
- Modify: `README.md` (test count)

- [ ] **Step 1: Run the full CPU suite**

Run: `python -m pytest -m "not gpu and not slow and not real_video" -q`
Expected: all PASS (the local Windows torch DLL may fail `test_pipeline_smoke`; that is a known environment issue — confirm it is the only failure and it is the torch `c10.dll` OSError, not a logic error. On PACE the suite is fully green.)

- [ ] **Step 2: Update `SETUP.md`** — rewrite §5 to describe the per-play tree +
  `meta.yaml` (use the spec's layout block), update §9 to the per-play DAG
  (no field stage; calibration is a manual per-play pre-step via
  `scripts/02_calibrate_cameras.py --play-dir ...`), and change the §10 stage CLI
  table to the `--play-dir PATH` invocation. Add a one-liner for
  `scripts/new_play.py`.

- [ ] **Step 3: Update `README.md`** — bump the test count to the new total
  (count with `python -m pytest -m "not gpu and not slow and not real_video"
  --collect-only -q | tail -1`).

- [ ] **Step 4: Commit**

```bash
git add SETUP.md README.md
git commit -m "Docs: per-play season tree (meta.yaml, --play-dir CLIs, per-play DAG)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push**

```bash
git push
```

---

## Self-Review (completed during planning)

- **Spec coverage:** directory layout → Task 1; meta.yaml → Task 2; discovery →
  Task 3; scaffolder → Task 4; stage CLIs `--play-dir` → Tasks 5-9; DAG fold +
  discovery → Task 10; calibration retarget + plays.py retirement + alignment →
  Task 11; `_library` reconciliation → Task 12; tests + docs → Tasks 12-13. All
  spec sections map to a task.
- **Type consistency:** `PlayDir` fields/properties (`dir`, `video`,
  `cameras_json`, `field_ply`, `tracks`, `entities`, `poses_dir`, `pose`, `ball`,
  `render_mp4`, `meta_yaml`, `library_root`, `rosters_root`, `registry_path`,
  `teams`, `from_dir`) are used consistently across Tasks 5-12. `PlayMeta`
  (`game_teams`, `fps`, `gsis_play_id`) consistent in Tasks 2, 7, 11.
  `scaffold_play(...)` signature matches its test (Task 4) and reuse (Task 11).
  `build_submission_plan(cfg, plays)` new signature consistent across Task 10.
- **Placeholders:** stage-CLI tasks (5-9) intentionally reuse each file's existing
  local helpers rather than reproduce unseen bodies; the exact API swap
  (`play_paths`/`load_plays` → `PlayDir.from_dir`/`load_meta`, `--play-dir`, drop
  window) is fully specified. No "TBD"/"handle edge cases" left.

## Known follow-ups (out of scope; do not implement here)
- Per-play SLURM array tasks need `perception_array.sbatch` / `render_array.sbatch`
  to read play #${SLURM_ARRAY_TASK_ID} from `outputs/play_worklist.txt` and pass
  `--play-dir`; finalize that sbatch plumbing at PACE bring-up alongside job-id
  capture in `--submit`.
- Semi-automated calibration (to avoid per-play manual clicking at season scale).
