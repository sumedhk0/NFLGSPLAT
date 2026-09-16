"""Triangulated hip ground points (render.tri_hips)."""
import numpy as np
import pandas as pd

from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
from nfl_gsplat.render import tri_hips as th


class _Track:
    def __init__(self, eye, target, n=5):
        R, t = look_at(np.asarray(eye, float), np.asarray(target, float))
        K = intrinsics(1920, 1080, fov_deg=12.0)
        self.K = np.repeat(K[None], n, axis=0)
        self.R = np.repeat(R[None], n, axis=0)
        self.t = np.repeat(t[None], n, axis=0)
        self.conf = np.ones(n)


def _project(track, f, X):
    p = track.R[f] @ np.asarray(X, float) + track.t[f]
    u = track.K[f] @ (p / p[2])
    return u[:2]


def _rows(cam, f, pid, uv, conf=0.9):
    return [dict(cam=cam, frame=f, global_player_id=pid, joint=j, x=uv[0] + dx, y=uv[1], conf=conf)
            for j, dx in ((11, -3.0), (12, 3.0))]


def test_two_hip_rays_meet_at_the_hip_and_the_gates_refuse_a_mispair():
    side = _Track(eye=(-3.6, 80.5, 35.9), target=(-20.0, 0.0, 1.0))
    end = _Track(eye=(-110.0, 0.5, 9.0), target=(-20.0, 0.0, 1.0))
    tracks = {"sideline": side, "endzone": end}
    X = np.array([-24.0, 3.0, 0.8])                       # a hip centre on the field
    Y = np.array([-30.0, -2.0, 0.9])                      # another man
    rows = _rows("sideline", 3, 1, _project(side, 0, X)) + _rows("endzone", 3, 1, _project(end, 0, X))
    rows += _rows("sideline", 3, 2, _project(side, 0, Y)) + _rows("endzone", 3, 2, _project(end, 0, X))  # id 2 mispaired
    rows += _rows("sideline", 4, 1, _project(side, 0, X), conf=0.1)                                       # low confidence
    kdf = pd.DataFrame(rows)
    tri = th.triangulated_hips(kdf, tracks)
    assert set(tri) == {(3, 1)}
    xy, z, gap = tri[(3, 1)]
    assert np.allclose(xy, X[:2], atol=1e-3) and abs(z - 0.8) < 1e-3 and gap < 1e-6
    # the mispaired id's rays miss each other by metres, or land at a wrong height: refused
    c1, d1 = th.pixel_ray(side.K[0], side.R[0], side.t[0], _project(side, 0, Y))
    c2, d2 = th.pixel_ray(end.K[0], end.R[0], end.t[0], _project(end, 0, X))
    P, g = th.closest_point(c1, d1, c2, d2)
    assert g > 0.5 or not (0.5 <= P[2] <= 1.4)
    # a frame shift on the endzone rows is honoured (the poses are identical here, so the point is the same)
    tri2 = th.triangulated_hips(kdf, tracks, frame_shift={"endzone": -1})
    assert (3, 1) in tri2 and np.allclose(tri2[(3, 1)][0], X[:2], atol=1e-3)
    # a frame beyond a camera's poses is skipped, not an error
    assert th.triangulated_hips(kdf.assign(frame=kdf.frame + 100), tracks) == {}
