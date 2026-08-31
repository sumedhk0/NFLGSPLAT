"""The jersey reader: number encoding, roster scoring, and that it can learn."""
import numpy as np
import pytest

from nfl_gsplat.identity.jersey_cnn import (
    BLANK,
    join_number,
    normalise,
    split_number,
)


def test_two_digit_numbers_split_and_rejoin():
    for n in (0, 7, 10, 42, 99):
        assert join_number(*split_number(n)) == n


def test_a_single_digit_number_uses_the_blank_tens_class():
    assert split_number(7) == (BLANK, 7)
    assert split_number(70) == (7, 0)


def test_out_of_range_numbers_are_refused():
    with pytest.raises(ValueError):
        split_number(100)


def test_normalise_puts_crops_in_channels_first_and_unit_range():
    crops = np.full((3, 64, 64, 3), 255, np.uint8)
    got = normalise(crops)
    assert got.shape == (3, 3, 64, 64)
    assert got.min() >= -1.0 and got.max() <= 1.0


def test_a_single_crop_is_accepted():
    assert normalise(np.zeros((64, 64, 3), np.uint8)).shape == (1, 3, 64, 64)


@pytest.mark.slow
def test_the_model_learns_a_separable_signal_and_scores_a_roster():
    """That the head wiring trains and scores a roster -- not an accuracy claim.

    The signal here is deliberately trivial and unambiguous: a solid block whose
    POSITION encodes the digit. Real accuracy is measured on real crops (and is
    poor -- see the module docstring); this only checks that the two heads learn
    and that number_logprobs ranks a roster.
    """
    pytest.importorskip("torch")
    from nfl_gsplat.identity.jersey_cnn import number_logprobs, train

    roster = [7, 12, 88]
    crops, numbers = [], []
    for n in roster:
        t, u = split_number(n)
        for _ in range(80):
            img = np.zeros((64, 64, 3), np.uint8)
            img[4:28, 2 + 5 * (t % 11):2 + 5 * (t % 11) + 20] = 255
            img[36:60, 2 + 5 * u:2 + 5 * u + 20] = 255
            crops.append(img)
            numbers.append(n)
    crops = np.stack(crops)
    numbers = np.asarray(numbers)

    model = train(crops, numbers, epochs=20, batch=64, lr=1e-3, device="cpu")
    scores = number_logprobs(model, crops[:12], roster, device="cpu")
    assert scores.shape == (12, len(roster))
    picked = np.array([roster[i] for i in scores.argmax(1)])
    # Chance on three classes is 4 of 12; require clearly better than that.
    assert (picked == numbers[:12]).sum() >= 8
