"""render.shade: kit colours lit by a stadium key light through the posed mesh's own normals."""
import numpy as np

from nfl_gsplat.render import shade


def _tetra():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])   # outward winding
    return verts, faces


def test_vertex_normals_point_out_of_the_body():
    verts, faces = _tetra()
    n = shade.vertex_normals(verts, faces)
    assert n.shape == (4, 3) and np.allclose(np.linalg.norm(n, axis=1), 1.0)
    centre = verts.mean(axis=0)
    assert ((verts - centre) * n).sum(axis=1).min() > 0              # every normal leaves the solid


def test_lit_faces_brighter_than_unlit_and_colour_kept_in_range():
    verts, faces = _tetra()
    colour = np.full((4, 3), 0.8)
    out = shade.shade(verts, faces, colour, light=(0.0, 0.0, 1.0), ambient=0.5, diffuse=0.6)
    assert out.shape == colour.shape and out.min() >= 0 and out.max() <= 1
    assert out[3].mean() > out[0].mean()                               # the top vertex faces the light, the corner does not
    same = shade.shade(verts, faces, colour, ambient=1.0, diffuse=0.0, bounce=0.0)
    assert np.allclose(same, colour)                                   # ambient 1, no diffuse, no bounce: the kit as it was


def test_lit_batch_lights_each_splat_through_its_own_normal():
    from nfl_gsplat.compositing.mesh_to_gaussians import _SH_C0, mesh_to_gaussians, vertex_normals
    verts, faces = _tetra()
    batch = mesh_to_gaussians(verts, faces, colour=(0.8, 0.8, 0.8))
    assert np.allclose(shade.splat_normals(batch.rot), vertex_normals(verts, faces), atol=1e-5)
    lit = shade.lit_batch(batch, light=(0.0, 0.0, 1.0), ambient=0.5, diffuse=0.6)
    rgb = lit.sh[:, :, 0] * _SH_C0 + 0.5
    assert rgb[3].mean() > rgb[0].mean() and rgb.min() >= 0 and rgb.max() <= 1
    assert np.array_equal(lit.xyz, batch.xyz) and np.array_equal(lit.rot, batch.rot)
    flat = shade.lit_batch(batch, ambient=1.0, diffuse=0.0, bounce=0.0)
    assert np.allclose(flat.sh, batch.sh, atol=1e-5)                  # no light: the batch as it was


def test_module_constants_are_read_at_call_time():
    """with_flag sets shade.AMBIENT after import: the default must follow it, or the A/B arm measures nothing."""
    verts, faces = _tetra()
    before = shade.shade(verts, faces, (0.5, 0.5, 0.5))
    old = shade.AMBIENT
    try:
        shade.AMBIENT = old + 0.2
        after = shade.shade(verts, faces, (0.5, 0.5, 0.5))
    finally:
        shade.AMBIENT = old
    assert np.allclose(after - before, 0.1)
