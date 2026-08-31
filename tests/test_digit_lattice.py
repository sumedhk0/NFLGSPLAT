"""Scoring reads against the roster rather than demanding exact matches."""
import collections

import pytest

from nfl_gsplat.identity.digit_lattice import (
    read_cost,
    read_weights,
    tally_lattice,
    unique_explanation,
)


def test_an_exact_read_costs_nothing():
    assert read_cost("41", "41") == 0.0


def test_losing_a_digit_is_cheaper_than_inventing_one():
    """OCR drops digits constantly and rarely hallucinates a whole extra one."""
    dropped = read_cost("4", "41")      # wore 41, read 4
    invented = read_cost("41", "4")     # wore 4, read 41
    assert dropped < invented


def test_a_shape_confusion_is_cheaper_than_an_arbitrary_one():
    assert read_cost("3", "8") < read_cost("3", "4")


def test_a_read_supports_the_roster_numbers_it_could_be():
    roster = [4, 41, 18, 90]
    w = read_weights("4", roster)
    assert w[4] > w[41]                 # exact beats truncated
    assert 41 in w                      # but 41 is still supported
    assert 90 not in w or w[90] < w[41]


def test_the_roster_disambiguates_an_ambiguous_string():
    """The whole reason this works: only 22 numbers are actually out there."""
    assert unique_explanation("4", [4, 18, 90, 55]) == 4
    # ...but not when two candidates are equally good.
    assert unique_explanation("4", [4, 41, 40]) in (4, None)


def test_tally_spreads_one_read_over_every_jersey_it_could_be():
    votes = {7: collections.Counter({"1": 3, "18": 5})}
    got = tally_lattice(votes, [1, 18, 81, 90])
    assert 18 in got[7] and 1 in got[7]
    # 18 was read outright five times and "1" also supports it
    assert got[7][18] > got[7][1]


def test_a_wild_read_is_dropped_rather_than_smeared():
    votes = {3: collections.Counter({"777": 4})}
    got = tally_lattice(votes, [1, 2, 3])
    assert 3 not in got or not got[3]


def test_evidence_accumulates_with_repetition():
    once = tally_lattice({1: collections.Counter({"90": 1})}, [90, 9])
    many = tally_lattice({1: collections.Counter({"90": 10})}, [90, 9])
    assert many[1][90] == pytest.approx(10 * once[1][90])
