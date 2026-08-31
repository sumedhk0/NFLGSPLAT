"""Score every OCR read against the whole roster, instead of only exact hits.

The reader returns a string of digits. Counting only the reads that exactly
match a jersey throws away most of what it saw: a glimpse of "4" on a player
wearing 41 is real evidence, and so is "18" read off an 18 that came back as
"13". Measured on this footage, of the tracks that got any read at all only 75%
had the right number as their top vote, while the read rate itself was 87% --
so the losses are in interpretation, not in seeing.

WHAT THE ERRORS ACTUALLY ARE, which is what the costs below encode:

  * A DIGIT GOES MISSING far more often than one is invented. Half a number is
    occluded by an arm, or falls off the crop. jersey_vote already measured
    this and gives truncations full credit: among contested tracks the top two
    candidates are digit-related 24-36% of the time against 6% for random
    pairs, and the confusions are exactly 14/4, 85/8, 13/1, 70/0.
  * DIGITS ARE CONFUSED IN PREDICTABLE PAIRS -- 8 with 3 and 0 and 6, 1 with 7,
    5 with 6. These are shape confusions, not random.

WHY THE ROSTER MAKES THIS WORK. Only 22 numbers are on the field. A read that
is ambiguous in the abstract is usually unambiguous against 22 candidates: "4"
narrows to whichever of 4, 40-49, 14, 24... are actually out there, and often
only one is. That is why this scores against the roster rather than trying to
decide what the string "really" said.

Output is deliberately the same shape jersey_vote.assign already consumes --
per track, a weight per jersey -- so it drops in as a parallel tally.

MEASURED, AND IT LOSES. Over 20 plays of cached reads it scored 48% recall
against 55% for plain exact matching, both at 100% precision. It is OFF by
default and kept because the reasoning above is sound and the failure is
informative: spreading a read across the numbers it could be also spreads it
onto the numbers it is not, and that compresses the winner's MARGIN. assign()
refuses thin margins by design, so evidence that makes every candidate a little
more plausible makes the assignment less willing to commit. Restricting reads
to the roster -- which the exact path already does -- turns out to do most of
the disambiguating work this was built to do.
"""
from __future__ import annotations

import collections
import math

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Cost of dropping a digit the player was really wearing. Cheap, because the
# reader loses digits constantly.
DELETION_COST: float = 1.0

# Cost of the reader inventing a digit that is not on the jersey. Expensive:
# OCR rarely hallucinates a whole extra digit next to a real one.
INSERTION_COST: float = 2.5

# Cost of reading one digit as another. The listed pairs are shape confusions
# and cost less than an arbitrary substitution.
SUBSTITUTION_COST: float = 2.0
CONFUSABLE_COST: float = 0.8
CONFUSABLE_PAIRS: frozenset = frozenset({
    frozenset("38"), frozenset("08"), frozenset("68"), frozenset("35"),
    frozenset("17"), frozenset("56"), frozenset("69"), frozenset("01"),
    frozenset("25"), frozenset("79"),
})

# Turns a cost into a weight. Larger makes the evidence flatter across
# candidates; smaller makes it nearly all-or-nothing on the exact match.
TAU: float = 1.0

# Beyond this cost a read says nothing about a jersey and is dropped, which
# keeps a wild read from spreading a thin smear of support over all 22.
MAX_COST: float = 4.0


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return (CONFUSABLE_COST if frozenset((a, b)) in CONFUSABLE_PAIRS
            else SUBSTITUTION_COST)


def read_cost(read: str, jersey: str) -> float:
    """Weighted edit distance from what was READ to what is WORN.

    Asymmetric on purpose: deleting a digit (the reader missed it) is cheap,
    inserting one (the reader invented it) is not.
    """
    n, m = len(read), len(jersey)
    # dp[i][j] = cost of explaining jersey[:j] given read[:i]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + INSERTION_COST      # read a digit not worn
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + DELETION_COST       # worn digit not read
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + _sub_cost(read[i - 1], jersey[j - 1]),
                dp[i - 1][j] + INSERTION_COST,
                dp[i][j - 1] + DELETION_COST,
            )
    return dp[n][m]


def read_weights(read: str, roster, *, tau: float = TAU,
                 max_cost: float = MAX_COST) -> dict[int, float]:
    """How much one read supports each jersey on the field."""
    out: dict[int, float] = {}
    for jersey in roster:
        cost = read_cost(str(read), str(int(jersey)))
        if cost <= max_cost:
            out[int(jersey)] = math.exp(-cost / tau)
    return out


def tally_lattice(votes_by_track, roster, *, tau: float = TAU,
                  max_cost: float = MAX_COST):
    """``{track: {jersey: weight}}`` from raw per-track read counters.

    ``votes_by_track`` maps track -> Counter(read -> times seen), exactly what
    ``jersey_ocr.read_jerseys`` returns. Every read contributes to every jersey
    it plausibly supports, weighted by how well it explains it.
    """
    cache: dict[str, dict[int, float]] = {}
    out: dict[int, collections.Counter] = {}
    roster = [int(j) for j in roster]
    for track, counter in votes_by_track.items():
        acc: collections.Counter = collections.Counter()
        for read, times in counter.items():
            key = str(read)
            if key not in cache:
                cache[key] = read_weights(key, roster, tau=tau,
                                          max_cost=max_cost)
            for jersey, weight in cache[key].items():
                acc[jersey] += weight * times
        if acc:
            out[int(track)] = acc
    _LOG.info("digit lattice: %d/%d tracks carry evidence",
              len(out), len(votes_by_track))
    return out


def unique_explanation(read: str, roster, *, margin: float = 2.0) -> int | None:
    """The jersey a read points to when it points at only ONE, else None.

    With 22 numbers on the field an ambiguous string is often unambiguous in
    context: a read of "4" means #4 only if no 4x, x4 or 4 is also out there.
    ``margin`` is how many times better the best explanation must be.
    """
    weights = read_weights(read, roster)
    if not weights:
        return None
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] >= margin * ranked[1][1]:
        return ranked[0][0]
    return None
