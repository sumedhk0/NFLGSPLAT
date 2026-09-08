"""identity.exclusive: one roster player, one avatar at a time."""
from nfl_gsplat.identity.exclusive import exclusive_names
from nfl_gsplat.identity.merge_cameras import PlayerIdentity


def _pi(jersey, player, team="KC"):
    return PlayerIdentity(jersey=jersey, player=player, team=team, height_m=1.85, weight_lb=0.0,
                          tracks={"sideline": 0, "endzone": 0})


def test_the_weaker_overlapping_claim_loses_the_name_and_a_tie_names_neither():
    merged = {1: _pi(7, "Kicker"), 2: _pi(7, "Kicker"), 3: _pi(10, "Runner"), 4: _pi(10, "Runner"),
              5: _pi(5, "Receiver"), 6: _pi(5, "Receiver"), 7: _pi(15, "Passer")}
    spans = {1: (0, 600), 2: (0, 460), 3: (0, 600), 4: (370, 640), 5: (500, 640), 6: (100, 300), 7: (0, 640)}
    evidence = {1: 8, 2: 3, 3: 6, 4: 5, 5: 4, 6: 7}
    out, demoted = exclusive_names(merged, spans, evidence)
    assert out[1].player == "Kicker" and out[2].player == "P2" and out[2].jersey == 0      # 8 vs 3: the weaker loses
    assert out[3].player == "P3" and out[4].player == "P4"                                  # 6 vs 5: a tie, neither
    assert out[5].player == "Receiver" and out[6].player == "Receiver"                      # no overlap: both stay
    assert out[7].player == "Passer"
    assert sorted(g for g, *_ in demoted) == [2, 3, 4]
    assert merged[2].player == "Kicker"                                                     # input untouched
