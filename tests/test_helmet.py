"""render.helmet: the head wears the team's shell."""
import numpy as np

from nfl_gsplat.render import helmet as hm


def _template():
    rng = np.random.default_rng(0)
    body = rng.uniform([-0.3, -1.0, -0.15], [0.3, 0.1, 0.15], size=(400, 3))     # torso and below
    head = np.array([0.0, 0.3, 0.0]) + 0.1 * rng.normal(size=(100, 3)) / np.sqrt(3)
    vt = np.vstack([body, head])
    joints = np.zeros((55, 3))
    joints[hm.NECK_JOINT] = (0.0, 0.11, 0.0)
    return vt, joints


def test_head_mask_is_the_vertices_above_the_neck():
    vt, joints = _template()
    m = hm.head_mask(vt, joints)
    assert m[400:].mean() > 0.9 and m[:400].sum() == 0


def test_wear_helmet_colours_and_inflates_only_the_head():
    vt, joints = _template()
    m = hm.head_mask(vt, joints)
    colours = np.full((len(vt), 3), 0.5)
    v2, c2 = hm.wear_helmet(vt, colours, m, hm.HELMET_RGB["KC"], inflate_m=0.02)
    assert np.allclose(c2[m], hm.HELMET_RGB["KC"]) and np.allclose(c2[~m], 0.5)
    assert np.allclose(v2[~m], vt[~m])
    centre = vt[m].mean(0)
    r_before = np.linalg.norm(vt[m] - centre, axis=1)
    r_after = np.linalg.norm(v2[m] - centre, axis=1)
    assert np.allclose(r_after - r_before, 0.02, atol=1e-6)
    assert np.allclose(vt, _template()[0])                     # inputs untouched


def test_single_colour_broadcasts():
    vt, joints = _template()
    m = hm.head_mask(vt, joints)
    v2, c2 = hm.wear_helmet(vt, (0.9, 0.9, 0.9), m, (0.1, 0.1, 0.1))
    assert c2.shape == vt.shape and np.allclose(c2[~m], 0.9) and np.allclose(c2[m], 0.1)
