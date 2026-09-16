"""08w: a weak kit vote is overruled by where the man lines up before the snap."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "teamform", Path(__file__).resolve().parents[1] / "scripts" / "08w_team_from_formation.py")
teamform = importlib.util.module_from_spec(_spec)
sys.modules["teamform"] = teamform
_spec.loader.exec_module(teamform)


def test_only_a_deep_weak_kit_defender_is_re_teamed():
    """Play 1's id 82: Baltimore by a 0.44 kit vote, 1 m inside the Kansas City line for the whole
    pre-snap. A defensive lineman on his own side, a wide receiver deep but wearing his kit plainly,
    and a deep man with a strong vote are all left alone."""
    los = {"x": -24.0, "sign": 1.0, "snap": 300}           # offence's side is x > -24
    ground = {f: {82: np.array([-23.0, 0.0]),               # 1.0 m on the offence's side, weak kit
                  15: np.array([-24.6, 1.0]),               # a defensive lineman 0.6 m on his own side
                  40: np.array([-18.0, 12.0]),              # a receiver split wide, 6 m deep, strong kit
                  17: np.array([-22.5, -1.0])} for f in range(180, 290)}
    kit_share = {82: 0.44, 15: 0.10, 40: 0.05, 17: 0.90}
    team_of = {82: "BAL", 15: "BAL", 40: "BAL", 17: "KC"}
    out = teamform.plan(ground, kit_share, team_of, los, depth_m=0.5, min_frames=15, weak=0.3, offence="KC", defence="BAL")
    assert [(p, t) for p, t, *_ in out] == [(82, "BAL")]
    pid, team, n_deep, dep, share = out[0]
    assert n_deep == 110 and abs(dep - 1.0) < 1e-9 and share == 0.44
    # too few deep frames: not moved
    few = {f: {82: np.array([-23.0, 0.0])} for f in range(180, 190)}
    assert teamform.plan(few, kit_share, team_of, los, depth_m=0.5, min_frames=15, weak=0.3, offence="KC", defence="BAL") == []
    # a strong vote is never overruled
    assert teamform.plan(ground, {**kit_share, 82: 0.05}, team_of, los, depth_m=0.5, min_frames=15, weak=0.3,
                         offence="KC", defence="BAL") == []


def test_positive_share_ignores_nan():
    import pandas as pd

    assert teamform.positive_share(pd.Series([0.2, -0.1, np.nan, 0.3])) == 2 / 3
    assert np.isnan(teamform.positive_share(pd.Series([np.nan])))
