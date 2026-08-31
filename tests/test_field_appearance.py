"""Sampling the venue's turf onto the spec field.

The rig is synthetic and the answer known: paint a pattern on the ground, look
at it with a camera, and check the pattern comes back where it started.
"""
import numpy as np
import pytest

from nfl_gsplat.field.field_appearance import (
    accumulate,
    composite,
    coverage_fraction,
    ground_grid,
    remap_points,
    sample_frame,
)

RES = 0.5                      # coarse, so the tests stay fast
EXTENT = (-30.0, 30.0, -20.0, 20.0)


def look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return R, -R @ centre


W, H = 640, 480
K = np.array([[500.0, 0, W / 2.0], [0, 500.0, H / 2.0], [0, 0, 1.0]])


def painted_ground():
    """A ground texture with a stripe pattern, in the map orientation."""
    mesh_x, mesh_y = ground_grid(RES, EXTENT)
    img = np.zeros((*mesh_x.shape, 3), np.uint8)
    img[..., 1] = 120                                   # turf green
    img[(np.floor(mesh_x / 5.0).astype(int) % 2) == 0, 1] = 180   # mow stripes
    img[np.abs(mesh_y) < 1.0] = (255, 255, 255)         # a painted line
    return img


def render_view(ground, centre):
    """Render the ground texture as a camera at ``centre`` would see it."""
    R, t = look_at(centre)
    x_min, x_max, y_min, y_max = EXTENT
    gh, gw = ground.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    pix = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    rays = (np.linalg.inv(K) @ pix.T).T @ R          # camera -> world dirs
    cam_c = -R.T @ t
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (0.0 - cam_c[2]) / rays[:, 2]
    hit = cam_c[None] + s[:, None] * rays
    col = (hit[:, 0] - x_min) / RES
    row = (y_max - hit[:, 1]) / RES
    ok = (s > 0) & (col >= 0) & (col < gw - 1) & (row >= 0) & (row < gh - 1)
    out = np.zeros((H * W, 3), np.uint8)
    out[ok] = remap_points(ground, col[ok], row[ok])
    return out.reshape(H, W, 3), R, t


def test_sample_frame_puts_the_pattern_back_where_it_came_from():
    ground = painted_ground()
    image, R, t = render_view(ground, (0.0, -45.0, 25.0))
    got, seen = sample_frame(image, K, R, t, res_m=RES, extent=EXTENT)
    assert seen.any()
    # Compare only where the camera actually saw turf.
    good = seen & (ground.sum(axis=2) > 0) & (got.sum(axis=2) > 0)
    assert good.sum() > 200
    diff = np.abs(got[good].astype(float) - ground[good].astype(float))
    assert np.median(diff) < 25.0


def test_unseen_texels_are_marked_not_guessed():
    ground = painted_ground()
    image, R, t = render_view(ground, (0.0, -45.0, 25.0))
    _got, seen = sample_frame(image, K, R, t, res_m=RES, extent=EXTENT)
    assert not seen.all()          # a narrow view cannot cover the whole field


def test_median_rejects_a_player_standing_on_the_turf():
    """The reason for a median: movers are a minority per texel, so they vanish."""
    ground = painted_ground()
    frames = []
    for i, cx in enumerate((-6.0, 0.0, 6.0, 12.0, -12.0)):
        image, R, t = render_view(ground, (0.0, -45.0, 25.0))
        # A magenta blob in a different place each frame, as a player would be.
        x0 = 200 + 40 * i
        image[240:300, x0:x0 + 60] = (255, 0, 255)
        frames.append((image, K, R, t))
    texture, coverage, counts = accumulate(frames, res_m=RES, extent=EXTENT,
                                           min_samples=3)
    assert coverage.any()
    magenta = ((texture[..., 0] > 200) & (texture[..., 1] < 80)
               & (texture[..., 2] > 200) & coverage)
    assert magenta.sum() == 0
    assert counts.max() == len(frames)


def test_texels_seen_too_few_times_are_not_trusted():
    ground = painted_ground()
    image, R, t = render_view(ground, (0.0, -45.0, 25.0))
    _tex, coverage, counts = accumulate([(image, K, R, t)], res_m=RES,
                                        extent=EXTENT, min_samples=3)
    assert not coverage.any()          # one frame can never reach three samples
    assert counts.max() == 1


def test_composite_shows_the_drawn_field_where_nothing_was_seen():
    ground = painted_ground()
    observed = np.full_like(ground, 200)
    coverage = np.zeros(ground.shape[:2], bool)
    coverage[:, :10] = True
    out = composite(observed, coverage, ground)
    assert (out[:, :10] == 200).all()
    assert (out[:, 10:] == ground[:, 10:]).all()


def test_composite_refuses_mismatched_textures():
    ground = painted_ground()
    with pytest.raises(ValueError):
        composite(np.zeros((4, 4, 3), np.uint8),
                  np.zeros((4, 4), bool), ground)


def test_coverage_fraction_is_a_fraction():
    m = np.zeros((10, 10), bool)
    m[:5] = True
    assert coverage_fraction(m) == pytest.approx(0.5)


def test_accumulate_refuses_an_empty_sequence():
    with pytest.raises(ValueError):
        accumulate([], res_m=RES, extent=EXTENT)


def test_grazing_rays_are_dropped_because_they_reach_the_crowd():
    """The ground plane is infinite; the ground is not.

    Near the horizon a ray passes just above the real field and lands on the
    stands, and the sampler will faithfully paint spectators onto the far
    sideline -- measured, a band of crowd across the top of a real texture.
    """
    ground = painted_ground()
    image, R, t = render_view(ground, (0.0, -45.0, 25.0))
    strict = sample_frame(image, K, R, t, res_m=RES, extent=EXTENT,
                          min_incidence_deg=25.0)[1]
    loose = sample_frame(image, K, R, t, res_m=RES, extent=EXTENT,
                         min_incidence_deg=0.0)[1]
    assert strict.sum() < loose.sum()
    # What survives is the near ground, not the far edge.
    rows = np.flatnonzero(strict.any(axis=1))
    assert rows.min() >= np.flatnonzero(loose.any(axis=1)).min()
