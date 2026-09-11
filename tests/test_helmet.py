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


def test_pads_mask_is_the_shoulders_and_they_broaden():
    rng = np.random.default_rng(1)
    body = rng.uniform([-0.1, -1.0, -0.1], [0.1, 0.4, 0.1], size=(300, 3))       # a narrow column
    shoulders = np.vstack([np.array([0.17, 0.09, 0.0]) + 0.05 * rng.normal(size=(40, 3)),
                           np.array([-0.17, 0.09, 0.0]) + 0.05 * rng.normal(size=(40, 3))])
    vt = np.vstack([body, shoulders])
    joints = np.zeros((55, 3))
    joints[16] = (0.17, 0.09, 0.0)
    joints[17] = (-0.17, 0.09, 0.0)
    m = hm.pads_mask(vt, joints)
    assert m[300:].mean() > 0.75 and m[:300].mean() < 0.15      # scatter puts a few past the radius
    # Placed upright: world z up. Use the template as if z were up for the width check.
    world = vt[:, [0, 2, 1]]                                   # x, (z->y), (y->z up)
    v2 = hm.wear_pads(world, m, out_m=0.035, up_m=0.02)
    width_before = np.ptp(world[m, 0])
    width_after = np.ptp(v2[m, 0])
    assert width_after > width_before + 0.05
    assert np.allclose(v2[~m], world[~m])
    assert (v2[m, 2] - world[m, 2]).min() > 0.019

