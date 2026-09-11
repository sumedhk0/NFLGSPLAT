"""One roster player is one avatar at a time.

WHY. 08c names each id on its own from its jersey vote, so two ids that
both read the same number carry the same name, and when their spans
overlap the render draws that player twice. Play 1 v16 (2026-09-08): four
names on two overlapping ids each -- "Harrison Butker" (#7, the kicker, on
a 2nd-and-20) on ids 16 and 17 for 616 frames, Isiah Pacheco on 5 and 37
for 278, Marquise Brown, Nelson Agholor. One of each pair is an OCR misread
(75 % per track).

WHAT. Among ids that claim the same (team, jersey) and overlap in time, the
one with the strongest evidence keeps the name; the others lose it (jersey
0, player "P<id>", the kit kept) rather than take a second guess. Evidence
is the number of crops that read the winning digit (jersey_votes_win),
then the number of frames the number was read on, then the span length;
a tie within ``TIE_RATIO`` names neither (two equal claims are one right
and one wrong, and a wrong name is the worse defect).
"""
from __future__ import annotations

from dataclasses import replace

TIE_RATIO: float = 1.5


def exclusive_names(merged: dict, spans: dict, evidence: dict, *, tie_ratio: float = TIE_RATIO):
    """``merged``: id -> PlayerIdentity (fields jersey, player, team);
    ``spans``: id -> (first frame, last frame); ``evidence``: id -> a number
    (bigger = stronger). Returns ``(merged, demoted)`` with ``demoted`` a list
    of (id, jersey, player, kept id or None)."""
    by_claim: dict = {}
    for gid, ident in merged.items():
        if ident.jersey > 0 and not str(ident.player).startswith("P"):
            by_claim.setdefault((ident.team, int(ident.jersey)), []).append(int(gid))
    out = dict(merged)
    demoted = []
    for claim, ids in by_claim.items():
        if len(ids) < 2:
            continue
        # cliques of overlapping ids: resolve pairwise, strongest first
        ids = sorted(ids, key=lambda g: (-float(evidence.get(g, 0.0)), g))
        kept: list = []
        for g in ids:
            lo, hi = spans.get(g, (None, None))
            rivals = [k for k in kept if spans.get(k, (None, None))[0] is not None and lo is not None
                      and not (hi < spans[k][0] or lo > spans[k][1])]
            if not rivals:
                kept.append(g)
                continue
            k = rivals[0]
            ev_g, ev_k = float(evidence.get(g, 0.0)), float(evidence.get(k, 0.0))
            if ev_k > 0 and ev_g > 0 and ev_k / ev_g < tie_ratio:
                # a tie: neither claim is sound
                for x in (g, k):
                    if x in out and not str(out[x].player).startswith("P"):
                        out[x] = replace(out[x], jersey=0, player=f"P{x}")
                        demoted.append((x, claim[1], merged[x].player, None))
                if k in kept:
                    kept.remove(k)
            else:
                out[g] = replace(out[g], jersey=0, player=f"P{g}")
                demoted.append((g, claim[1], merged[g].player, k))
    return out, demoted


SPECIALISTS = ("K", "P", "LS")


def specialist_veto(position, *, kicking_play: bool = False) -> bool:
    """A kicker, punter or long snapper is not on the field on a scrimmage
    down: a number that lands on one is a misread (play 1: two endzone
    tracks read "7" = the kicker on a 2nd-and-20). Off on a kicking play."""
    return (not kicking_play) and str(position).upper() in SPECIALISTS
