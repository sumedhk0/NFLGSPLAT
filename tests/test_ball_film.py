import numpy as np

from nfl_gsplat.tracking import ball_film as bf


def _flight_cands(f0=527, f1=562, x0=1350.0, y0=550.0, vx=-13.0, vy=-1.2, seed=0, holes=(541, 542, 555, 556)):
    rng = np.random.default_rng(seed)
    cands = {}
    for f in range(f0 - 6, f1 + 8):
        cs = []
        if f0 + 8 <= f <= f1 - 4 and f not in holes:            # the open-field stretch of the flight
            cs.append((x0 + vx * (f - f0) + rng.normal(0, 2), y0 + vy * (f - f0) + rng.normal(0, 2), 30))
            cs.append((x0 + vx * (f - 1 - f0), y0 + vy * (f - 1 - f0), 28))   # the previous frame's ghost
        cs.append((117.0, 123.0, 12))                            # a static graphic
        if f % 5 == 0:
            cs.append((rng.uniform(0, 1900), rng.uniform(150, 1000), 20))   # junk
        cands[f] = cs
    return cands


def test_fit_flight_finds_the_line_through_the_moving_blob_and_ignores_the_static_and_junk():
    fl = bf.fit_flight(_flight_cands())
    assert fl is not None
    assert 535 <= fl["frames"][0] <= 536 and 557 <= fl["frames"][-1] <= 558
    assert abs(fl["speed"] - 13.05) < 1.5
    x, y = bf.track_at(fl, 562)
    assert abs(x - (1350 - 13 * 35)) < 8 and abs(y - (550 - 1.2 * 35)) < 8
    assert bf.fit_flight({f: [(117.0, 123.0, 12)] for f in range(500, 560)}) is None


def _boxes():
    boxes = {}
    for f in range(490, 580):
        boxes[f] = [(80, 1320.0, 470.0, 1400.0, 600.0),          # the passer, still
                    (74, 880.0 - 5.0 * (f - 560), 480.0, 940.0 - 5.0 * (f - 560), 610.0),   # the receiver running left
                    (77, 300.0, 60.0, 360.0, 180.0)]              # the far-sideline runner
    return boxes


def test_a_longer_track_with_no_box_at_either_end_loses_to_the_pass():
    cands = _flight_cands()
    for f in range(495, 525):                                   # an unboxed man running in the open, 30 frames straight
        cands.setdefault(f, []).append((200.0 + 14.0 * (f - 495), 380.0 - 6.0 * (f - 495), 40))
    fls = bf.candidate_flights(cands)
    assert fls[0]["frames"][0] == 495 and fls[0]["n"] >= 25       # by inliers alone the runner wins ...
    fl = bf.fit_flight(cands, _boxes())
    assert 535 <= fl["frames"][0] <= 536                           # ... with the boxes the pass does
    assert bf.fit_flight(cands)["frames"] == fls[0]["frames"]
    # a line through blobs far apart in time with nothing between is not a flight
    sparse = {f: [(100.0 + 10.0 * (f - 500), 300.0, 20)] for f in (500, 501, 502, 580, 581, 582)}
    assert bf.candidate_flights(sparse) == []
    assert not bf.dense_enough([500, 501, 502, 580, 581, 582]) and bf.dense_enough([539, 540, 550, 551, 553, 554, 557, 558])


def test_name_ends_walks_into_the_passers_and_the_receivers_boxes():
    fl = bf.fit_flight(_flight_cands())
    boxes = _boxes()
    ends = bf.name_ends(fl, boxes, reach=20, pad=4.0)
    assert ends["passer"] == 80 and 526 <= ends["release"] <= 532
    assert ends["receiver"] == 74 and 557 <= ends["catch"] <= 566
    pid, d = bf.nearest_box(boxes[562], *bf.track_at(fl, 562))
    assert pid == 74 and d < 60


def test_a_short_wander_between_two_boxes_loses_to_the_long_flight_and_alone_is_no_flight():
    cands = _flight_cands()
    for f in range(520, 545):                                   # a blob drifting 60 px inside the pocket, 25 frames, 2.5 px/frame
        cands.setdefault(f, []).append((1330.0 + 2.4 * (f - 520), 700.0 + 0.5 * (f - 520), 15))
    boxes = _boxes()
    for f in boxes:
        boxes[f].append((197, 1300.0, 620.0, 1345.0, 720.0))    # a lineman's box the wander starts in ...
        boxes[f].append((79, 1380.0, 660.0, 1430.0, 760.0))     # ... and the quarterback's it ends in
    fl = bf.fit_flight(cands, boxes)
    assert 535 <= fl["frames"][0] <= 536                           # the real flight wins
    assert bf.track_length(fl) > 250
    short = {f: [(1330.0 + 2.4 * (f - 520), 700.0 + 0.5 * (f - 520), 15)] for f in range(520, 545)}
    assert bf.fit_flight(short, boxes) is None                     # too slow and too short to be a flight at all
    fast_short = {f: [(1330.0 + 7.0 * (f - 520), 700.0, 15)] for f in range(520, 530)}    # 63 px at 7 px/frame
    assert bf.fit_flight(fast_short, boxes) is None                # under MIN_LENGTH_PX ...
    assert bf.fit_flight(fast_short, boxes, min_length=10.0) is not None   # ... and a flight when the floor is lowered


def test_the_receiver_is_the_passers_teammate_when_a_defender_holds_the_same_point():
    fl = bf.fit_flight(_flight_cands())
    boxes = _boxes()
    for f in boxes:                                              # a defender draped on the receiver, a smaller box
        boxes[f].append((55, 890.0 - 5.0 * (f - 560), 500.0, 935.0 - 5.0 * (f - 560), 600.0))
    teams = {80: "KC", 74: "KC", 77: "KC", 55: "BAL"}
    ends = bf.name_ends(fl, boxes, teams=teams)
    assert ends["passer"] == 80 and ends["receiver"] == 74 and ends["others"] == [55]
    assert bf.name_ends(fl, boxes)["receiver"] == 55            # without teams the smaller box wins: the ambiguity is real
