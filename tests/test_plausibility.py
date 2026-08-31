"""Does the flailing detector catch injected faults and leave good motion alone?

The faults are the ones actually seen in the output: a limb snapping to a wrong
place for a frame or two, a joint teleporting, and NaNs from a failed fit.
"""
import numpy as np
import pytest

from nfl_gsplat.pose.forward_kinematics import NUM_BODY_JOINTS, SMPLX_BODY_PARENTS
from nfl_gsplat.pose.plausibility import (
    audit,
    bone_lengths,
    reference_bone_lengths,
    repair,
)

FPS = 59.94


def clean_sequence(T=120, seed=0):
    """A rigid skeleton translating and rotating smoothly -- bones never change."""
    rng = np.random.default_rng(seed)
    parents = np.asarray(SMPLX_BODY_PARENTS)
    # Build one body: each joint a fixed offset from its parent.
    offsets = rng.normal(0.0, 0.18, size=(NUM_BODY_JOINTS, 3))
    offsets[0] = 0.0
    rest = np.zeros((NUM_BODY_JOINTS, 3))
    for j in range(1, NUM_BODY_JOINTS):
        rest[j] = rest[parents[j]] + offsets[j]

    t = np.arange(T) / FPS
    seq = np.empty((T, NUM_BODY_JOINTS, 3))
    for i, ti in enumerate(t):
        ang = 0.6 * ti
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        seq[i] = rest @ R.T + np.array([4.0 * ti, 0.5 * ti, 1.0])
    return seq


def test_clean_motion_is_accepted():
    got = audit(clean_sequence(), fps=FPS)
    assert got.n_bad == 0
    assert np.nanmax(got.bone_error) < 1e-9


def test_bone_lengths_are_constant_for_a_rigid_body():
    lens = bone_lengths(clean_sequence())
    assert lens.shape == (120, NUM_BODY_JOINTS - 1)
    assert np.allclose(lens.std(axis=0), 0.0, atol=1e-9)


def test_reference_uses_the_median_so_bad_frames_cannot_move_it():
    seq = clean_sequence()
    truth = reference_bone_lengths(seq)
    seq[40:48, 7] += 3.0          # a limb thrown far away for 8 frames
    assert np.allclose(reference_bone_lengths(seq), truth, atol=1e-9)


def test_a_dislocated_limb_is_caught():
    """Every injected frame is rejected, and the snap BACK is rejected too.

    The frame where the limb returns is also implausible -- it carries the same
    impossible acceleration as the frame that threw it out -- so the detector
    flags one more frame than was injected. That is right, not slack, and the
    test asserts it rather than tolerating it.
    """
    seq = clean_sequence()
    seq[60:63, 5] += 0.9          # one joint 90 cm out of place
    got = audit(seq, fps=FPS)
    assert not got.ok[60:63].any()
    assert "bone(s) off" in got.reasons[61]
    assert got.ok[59] and got.ok[66]
    assert got.n_bad <= 5         # the three injected, plus the return


def test_a_teleport_is_caught_by_speed():
    seq = clean_sequence()
    seq[80] += 5.0                # the whole body jumps for one frame
    got = audit(seq, fps=FPS)
    assert not got.ok[80]
    assert "speed" in got.reasons[80] or "accel" in got.reasons[80]


def test_non_finite_frames_are_caught():
    seq = clean_sequence()
    seq[10, 3] = np.nan
    got = audit(seq, fps=FPS)
    assert not got.ok[10]
    assert "non-finite" in got.reasons[10]


def test_ground_penetration_is_only_checked_when_asked():
    seq = clean_sequence()
    seq[:, :, 2] -= 2.0           # sink the whole body below the turf
    assert audit(seq, fps=FPS).n_bad == 0
    sunk = audit(seq, fps=FPS, ground_z=0.0)
    assert sunk.n_bad == len(seq)
    assert "below the turf" in sunk.reasons[0]


def test_repair_fills_a_short_run_and_leaves_a_long_one():
    seq = clean_sequence(T=160)
    truth = seq.copy()
    seq[50:54, 5] += 0.9          # 4 bad frames -- fillable
    seq[100:130, 5] += 0.9        # 30 bad frames -- too long to invent
    got = audit(seq, fps=FPS)
    fixed, valid = repair(seq, got, max_gap=8)
    # The short run is filled with something continuous, not left as a hole.
    assert valid[50:54].all()
    assert np.isfinite(fixed[50:54]).all()
    assert np.abs(fixed[50:54, 5] - truth[50:54, 5]).max() < 0.2
    # The long run is NOT invented.
    assert not valid[105:125].any()
    assert np.isnan(fixed[105:125]).all()


def test_audit_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        audit(np.zeros((10, 3)))


def test_summary_reports_what_was_rejected():
    seq = clean_sequence()
    seq[60:63, 5] += 0.9
    got = audit(seq, fps=FPS)
    assert "/120 frames implausible" in got.summary()
    # The three injected frames, plus the frames where the limb snaps back --
    # which carry the same impossible acceleration and are correctly rejected.
    assert not got.ok[60:63].any()
    assert 3 <= got.n_bad <= 5


def test_a_single_dislocated_joint_is_caught_however_few_bones_it_moves():
    """The case a fraction-of-the-skeleton rule is blind to.

    Joint 5 has one parent bone and one child bone, so throwing it out of place
    disturbs 2 of the body's 21 bones -- under 10%. An earlier fraction rule
    missed all thirty frames of this while the worst bone read 737% too long.
    """
    seq = clean_sequence(T=160)
    seq[100:130, 5] += 0.9
    got = audit(seq, fps=FPS)
    assert got.ok[:99].all()
    assert not got.ok[100:130].any()
    assert got.bone_error[110] > 1.0


def test_mark_implausible_strikes_out_bad_frames_for_the_fit():
    from nfl_gsplat.pose.plausibility import mark_implausible

    seq = clean_sequence(T=120)
    seq[70:74, 5] += 0.9
    valid = np.ones((120, NUM_BODY_JOINTS), bool)
    got_valid, report = mark_implausible(seq, valid, fps=FPS)
    assert not got_valid[70:74].any()      # struck out across ALL joints
    assert got_valid[:69].all()
    assert report.n_bad >= 4


def test_a_short_bone_cannot_convict_a_frame_on_relative_error_alone():
    """The defect that rejected 100% of real frames.

    The collar and spine links are a few centimetres long, so a millimetre of
    fit noise is tens of percent. Judged on relative error alone the worst bone
    is always one of those, and every frame is condemned. A bone has to be
    wrong in metres as well.
    """
    seq = clean_sequence(T=60)
    parents = np.asarray(SMPLX_BODY_PARENTS)
    child = np.flatnonzero(parents >= 0)
    # Find the shortest bone and disturb it by a large FRACTION but a tiny
    # distance -- exactly the real failure mode.
    lens = bone_lengths(seq)[0]
    shortest = child[int(np.argmin(lens))]
    seq[30:34, shortest] += 0.30 * lens.min()      # 30% of a very short bone
    got = audit(seq, fps=FPS)
    assert got.ok[30:34].all()


def test_a_real_bone_error_is_still_caught():
    seq = clean_sequence(T=60)
    seq[30:34, 5] += 0.5                            # half a metre out of place
    got = audit(seq, fps=FPS)
    assert not got.ok[30:34].any()


def test_gaps_in_a_track_are_not_read_as_teleports():
    """Players are not detected every frame, and the audit must know that.

    Measured on real output: a track whose three surviving frames spanned video
    frames 48 to 140 was rejected for a 46 m/s joint speed that never happened,
    because consecutive ENTRIES were assumed to be consecutive FRAMES.
    """
    seq = clean_sequence(T=200)
    keep = [0, 1, 2, 90, 91, 92, 180, 181, 182]
    sparse = seq[keep]
    naive = audit(sparse, fps=FPS)                      # assumes 1/fps spacing
    aware = audit(sparse, fps=FPS, frame_indices=keep)  # knows the real gaps
    assert aware.n_bad <= naive.n_bad
    assert aware.n_bad == 0
