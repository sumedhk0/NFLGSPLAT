"""render.uniform: a kit painted by body region."""
import numpy as np

from nfl_gsplat.render import uniform as un


def _template():
    """A stick-figure template in SMPL-X's frame (y up, x across the shoulders)."""
    rng = np.random.default_rng(0)
    torso = rng.uniform([-0.15, -0.33, -0.1], [0.15, 0.10, 0.1], size=(200, 3))
    head = np.array([0.0, 0.30, 0.0]) + 0.08 * rng.normal(size=(60, 3))
    arms = np.vstack([rng.uniform([0.17, 0.0, -0.05], [0.70, 0.09, 0.05], size=(80, 3)),
                      rng.uniform([-0.70, 0.0, -0.05], [-0.17, 0.09, 0.05], size=(80, 3))])
    legs = rng.uniform([-0.15, -1.30, -0.1], [0.15, -0.36, 0.1], size=(200, 3))
    vt = np.vstack([torso, head, arms, legs])
    J = np.zeros((55, 3))
    J[un.PELVIS_JOINT] = (0.0, -0.35, 0.0)
    J[un.NECK_JOINT] = (0.0, 0.11, 0.0)
    J[4] = (0.11, -0.82, 0.0)
    J[5] = (-0.11, -0.82, 0.0)
    J[7] = (0.07, -1.23, 0.0)
    J[8] = (-0.07, -1.23, 0.0)
    J[18] = (0.42, 0.02, 0.0)
    J[19] = (-0.42, 0.04, 0.0)
    J[20] = (0.67, 0.04, 0.0)
    J[21] = (-0.67, 0.04, 0.0)
    return vt, J


def test_regions_partition_the_body():
    vt, J = _template()
    masks = un.regions(vt, J)
    total = np.zeros(len(vt), int)
    for m in masks.values():
        total += m.astype(int)
    assert (total == 1).all()                                   # exactly one region each
    assert all(m.sum() > 0 for m in masks.values()), {k: int(m.sum()) for k, m in masks.items()}
    assert vt[masks["jersey"], 1].min() > vt[masks["pants"], 1].max() - 1e-9
    assert vt[masks["pants"], 1].min() > vt[masks["socks"], 1].max() - 1e-9
    assert np.abs(vt[masks["gloves"], 0]).min() > np.abs(vt[masks["skin"], 0]).max() - 1e-9
    assert vt[masks["helmet"], 1].min() > vt[masks["jersey"], 1].max() - 1e-9


def test_dress_paints_the_kit():
    vt, J = _template()
    masks = un.regions(vt, J)
    c = un.dress(masks, un.KITS["KC"])
    assert c.shape == (len(vt), 3)
    assert np.allclose(c[masks["jersey"]], un.KITS["KC"].jersey)
    assert np.allclose(c[masks["pants"]], un.KITS["KC"].pants)
    assert np.allclose(c[masks["skin"]], un.SKIN_RGB)
    d = un.dress(masks, un.KITS["BAL"])
    assert not np.allclose(c[masks["jersey"]], d[masks["jersey"]])


def test_number_lands_on_the_jersey_back_and_front():
    rng = np.random.default_rng(1)
    vt, J = _template()
    # give the torso a front and a back
    vt = vt.copy()
    vt[:200, 2] = rng.choice([-0.1, 0.1], size=200)
    masks = un.regions(vt, J)
    base = un.dress(masks, un.KITS["KC"])
    out = un.paint_number(base, vt, J, masks, 15, un.KITS["KC"].number)
    changed = np.any(out != base, axis=1)
    assert changed.sum() > 10
    assert not changed[~masks["jersey"]].any()                        # only the jersey
    assert (vt[changed, 2] < 0).any() and (vt[changed, 2] > 0).any()  # back and front
    assert (vt[changed, 1] <= J[un.NECK_JOINT, 1] - un.NUMBER_TOP_M + 1e-9).all()
    ink = un.number_raster(88)
    assert 0.1 < ink.mean() < 0.6

