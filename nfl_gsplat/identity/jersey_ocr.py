"""Read jersey numbers per TRACK, and split tracks by team colour.

Two things gate identity, and neither is the OCR model:

* **Sampling budget.** A single crop reads correctly maybe 10% of the time, so a
  track's identity is a voting problem. Sampling 25 crops from each of 12 tracks
  gave 8 usable identities; the same OCR over 90 crops gave 14. The evidence is
  there, it just has to be collected.
* **Candidate set.** With the 22 known players from participation, a read of a
  number nobody is wearing is a misread and can be discarded outright. Team
  colour halves what remains -- an Arizona track cannot be a Seattle player, no
  matter what the OCR says.

Colour is clustered rather than hard-coded. Two teams on a field separate
cleanly in HSV whatever their uniforms, and hand-specifying colours per matchup
would need maintaining for every team pair and every alternate jersey.
"""
from __future__ import annotations

import collections

import cv2
import numpy as np

from nfl_gsplat.identity.team_color import dominant_jersey_color, split_two_teams
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Numbers sit on the upper back and chest. Cropping the whole body hands the
# reader legs, turf and other players; this band is where the digits are.
_TORSO_TOP: float = 0.18
_TORSO_BOTTOM: float = 0.55
_UPSCALE: float = 4.0
_MIN_BOX_H: int = 40
_MIN_CONF: float = 0.30


def plan_samples(tracks_df, *, min_track_len: int = 100, per_track: int = 120):
    """``{frame: [rows]}`` -- which crops to read, spread across each track.

    Spread rather than consecutive: neighbouring frames show the same pose from
    the same angle, so a hundred of them carry barely more information than one.
    A number hidden at the snap may be square to the camera a second later.
    """
    lengths = tracks_df.groupby("track_id").size()
    keep = [t for t in lengths.index if t >= 0 and lengths[t] >= min_track_len]
    wanted: dict[int, list] = collections.defaultdict(list)
    for track_id in keep:
        rows = tracks_df[tracks_df.track_id == track_id]
        step = max(1, len(rows) // per_track)
        for row in rows.iloc[::step][:per_track].itertuples():
            wanted[int(row.frame)].append(row)
    _LOG.info("jersey OCR: %d tracks >= %d frames, %d crops across %d frames",
              len(keep), min_track_len, sum(len(v) for v in wanted.values()),
              len(wanted))
    return dict(wanted), keep


def read_jerseys(frame_iter, wanted, *, reader=None, gpu: bool = True,
                 min_conf: float = _MIN_CONF, min_box_h: int = _MIN_BOX_H,
                 upscale: float = _UPSCALE):
    """One pass over the video. Returns ``(votes, colours)`` per track.

    ``votes`` maps track -> Counter(jersey -> count); ``colours`` maps track ->
    mean torso HSV, used downstream to split the two teams.

    ``min_conf``, ``min_box_h`` and ``upscale`` are the yield knobs. They are
    parameters rather than constants because the right values differ per feed by
    an order of magnitude: the endzone camera is zoomed roughly ten times
    tighter than the sideline, and read jerseys five to ten times better on the
    same play. A floor tuned for one feed throws away the other's evidence.

    Lowering ``min_conf`` is safer than it looks. Downstream,
    :func:`~nfl_gsplat.identity.jersey_vote.restrict_to_known` discards any read
    of a number nobody on the field is wearing, and
    :func:`~nfl_gsplat.identity.jersey_vote.credit_truncations` absorbs the
    commonest remaining error, so extra low-confidence reads mostly land as
    noise that is filtered rather than as wrong identities.
    """
    if reader is None:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    votes: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    colours: dict[int, list] = collections.defaultdict(list)

    for idx, bgr in frame_iter:
        for row in wanted.get(int(idx), ()):
            x1, y1 = int(max(0, row.bbox_x1)), int(max(0, row.bbox_y1))
            x2 = int(min(bgr.shape[1], row.bbox_x2))
            y2 = int(min(bgr.shape[0], row.bbox_y2))
            if y2 - y1 < min_box_h or x2 - x1 < 10:
                continue
            body = bgr[y1:y2, x1:x2]
            if body.size:
                colours[int(row.track_id)].append(dominant_jersey_color(body))

            height = y2 - y1
            crop = bgr[y1 + int(_TORSO_TOP * height):y1 + int(_TORSO_BOTTOM * height),
                       x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                              interpolation=cv2.INTER_CUBIC)
            for _box, text, conf in reader.readtext(crop, allowlist="0123456789"):
                digits = "".join(ch for ch in text if ch.isdigit())
                if digits and conf >= min_conf:
                    votes[int(row.track_id)][int(digits)] += 1

    mean_colours = {t: np.mean(np.stack(c), axis=0) for t, c in colours.items() if c}
    return dict(votes), mean_colours


def split_by_team(colours, votes, on_field):
    """``{track_id: team}`` by clustering torso colour, labelled via the roster.

    The clusters themselves are just "group 0" and "group 1"; which is Arizona
    is decided by whichever team's jerseys its members actually read as. That
    avoids hard-coding uniform colours per matchup, and it fails visibly rather
    than silently -- a cluster whose reads belong to neither team gets no label.
    """
    tracks = sorted(colours)
    if len(tracks) < 2:
        return {}
    labels = split_two_teams(np.stack([colours[t] for t in tracks]))

    jersey_team = {int(r.jersey_number): r.team for r in on_field.itertuples()}
    tally = {0: collections.Counter(), 1: collections.Counter()}
    for track_id, label in zip(tracks, labels):
        for jersey, count in votes.get(track_id, {}).items():
            team = jersey_team.get(int(jersey))
            if team:
                tally[int(label)][team] += count

    assignment = {}
    for label in (0, 1):
        if tally[label]:
            assignment[label] = tally[label].most_common(1)[0][0]
    if len(set(assignment.values())) < len(assignment):
        _LOG.warning("both colour clusters voted for the same team (%s); "
                     "colour is not separating these uniforms, so no team "
                     "constraint is applied", assignment)
        return {}
    out = {t: assignment[int(lab)] for t, lab in zip(tracks, labels)
           if int(lab) in assignment}
    _LOG.info("team split: %s", collections.Counter(out.values()))
    return out
