"""Calibration tests — the first gate. Everything downstream depends on this.

- solve_pnp on synthetic annotations → RMS < 1 px (pixel-exact synth expects ~0).
- Recovered (R, t) match ground truth within tight tolerance.
- Too-few-landmarks case raises CalibrationError.
- Missing annotations file raises SetupError.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nfl_gsplat.calibration.field_landmarks import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HASH_OFFSET_M,
    NFL_LANDMARKS,
    _yardline_x_m,
    landmark_points,
    list_landmark_names,
)
from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_annotations
from nfl_gsplat.errors import CalibrationError, SetupError
from nfl_gsplat.utils.io import read_json, write_json
from tests.fixtures.generate import (
    FIXTURE_HEIGHT,
    FIXTURE_WIDTH,
    generate,
)


# --- Landmark sanity --------------------------------------------------------

def test_field_dimensions_match_rulebook():
    # 120 yd length, 53⅓ yd width (with margin for float rounding).
    assert abs(FIELD_LENGTH_M - 109.728) < 1e-3
    assert abs(FIELD_WIDTH_M - 48.768) < 1e-3
    # Hash offset from centerline: 80 ft − 70.75 ft = 9.25 ft = 2.8194 m
    assert abs(HASH_OFFSET_M - 2.8194) < 1e-3


def test_yardline_positions():
    assert _yardline_x_m("mid_50") == 0.0
    assert _yardline_x_m("home_goal") == pytest.approx(45.720, abs=1e-3)
    assert _yardline_x_m("away_goal") == pytest.approx(-45.720, abs=1e-3)
    # 35 yd line on home side = 35 yd from home goal = 15 yd from 50.
    assert _yardline_x_m("home_35") == pytest.approx(15 * 0.9144, abs=1e-3)
    assert _yardline_x_m("away_25") == pytest.approx(-(25 * 0.9144), abs=1e-3)


def test_every_landmark_is_on_field():
    # Every defined landmark is within the field rectangle (+ small epsilon).
    eps = 1e-6
    hx = FIELD_LENGTH_M / 2.0
    hy = FIELD_WIDTH_M / 2.0
    for name, xyz in NFL_LANDMARKS.items():
        assert abs(xyz[0]) <= hx + eps, name
        assert abs(xyz[1]) <= hy + eps, name
        assert xyz[2] == 0.0, name


def test_landmark_points_stack_shape():
    names = ["mid_50_left_sideline", "home_goalpost_base", "away_25_right_hash"]
    pts = landmark_points(names)
    assert pts.shape == (3, 3)
    assert pts.dtype == np.float64


def test_landmark_points_bad_name():
    with pytest.raises(KeyError):
        landmark_points(["not_a_real_landmark"])


def test_list_landmark_names_nonempty():
    names = list_landmark_names()
    assert len(names) > 50  # 21 yard lines * 4 intersections + endzone + pylons
    assert sorted(names) == names  # sorted


# --- PnP on synthetic fixture ----------------------------------------------

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("nfl_fixture")
    generate(out)
    return out


def test_calibration_recovers_sideline_camera(fixture_dir: Path):
    ann = fixture_dir / "sideline_landmarks.json"
    gt = read_json(fixture_dir / "cameras_gt.json")["sideline"]
    res = solve_pnp_from_annotations(
        ann,
        image_size=(FIXTURE_WIDTH, FIXTURE_HEIGHT),
        max_reproj_px=1.0,
        bundle_adjustment=True,
    )
    # Synthetic input is pixel-exact → residuals should be essentially zero.
    assert res.rms_px < 0.5, f"rms={res.rms_px} too high"

    R_gt = np.asarray(gt["R"])
    t_gt = np.asarray(gt["t"])
    # Rotation alignment — Frobenius norm should be tiny.
    assert np.linalg.norm(res.pose.R - R_gt) < 5e-2
    # Camera center in world frame; tolerate small drift.
    c_recovered = res.pose.center_world()
    c_gt = -R_gt.T @ t_gt
    assert np.linalg.norm(c_recovered - c_gt) < 0.2, "camera center drifted >20 cm"


def test_calibration_recovers_endzone_camera(fixture_dir: Path):
    ann = fixture_dir / "endzone_landmarks.json"
    gt = read_json(fixture_dir / "cameras_gt.json")["endzone"]
    res = solve_pnp_from_annotations(
        ann,
        image_size=(FIXTURE_WIDTH, FIXTURE_HEIGHT),
        max_reproj_px=1.0,
        bundle_adjustment=True,
    )
    assert res.rms_px < 0.5

    R_gt = np.asarray(gt["R"])
    t_gt = np.asarray(gt["t"])
    assert np.linalg.norm(res.pose.R - R_gt) < 5e-2
    c_recovered = res.pose.center_world()
    c_gt = -R_gt.T @ t_gt
    assert np.linalg.norm(c_recovered - c_gt) < 0.2


def test_calibration_rejects_too_few_landmarks(fixture_dir: Path, tmp_path: Path):
    full = read_json(fixture_dir / "sideline_landmarks.json")
    truncated = full[:3]  # only 3 annotations → below min_landmarks=6
    ann = tmp_path / "too_few.json"
    write_json(ann, truncated)
    with pytest.raises(CalibrationError, match="need ≥"):
        solve_pnp_from_annotations(
            ann,
            image_size=(FIXTURE_WIDTH, FIXTURE_HEIGHT),
            max_reproj_px=5.0,
        )


def test_calibration_rejects_high_reprojection(fixture_dir: Path, tmp_path: Path):
    # Perturb annotations by 30 px so no solver can meet a 1 px threshold.
    full = read_json(fixture_dir / "sideline_landmarks.json")
    rng = np.random.default_rng(42)
    noisy = [
        {"name": e["name"],
         "uv": [e["uv"][0] + rng.uniform(20, 30), e["uv"][1] + rng.uniform(-30, -20)],
         "frame": e["frame"]}
        for e in full
    ]
    ann = tmp_path / "noisy.json"
    write_json(ann, noisy)
    with pytest.raises(CalibrationError, match="exceeds threshold"):
        solve_pnp_from_annotations(
            ann,
            image_size=(FIXTURE_WIDTH, FIXTURE_HEIGHT),
            max_reproj_px=1.0,
        )


def test_missing_annotations_file_is_setup_error(tmp_path: Path):
    with pytest.raises(SetupError, match="SETUP.md"):
        solve_pnp_from_annotations(
            tmp_path / "does_not_exist.json",
            image_size=(FIXTURE_WIDTH, FIXTURE_HEIGHT),
        )
