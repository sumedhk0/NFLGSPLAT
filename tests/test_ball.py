"""The football's mesh: size, orientation along the velocity, colours."""
import numpy as np

from nfl_gsplat.render import ball


def test_ball_mesh_is_a_prolate_ellipsoid_pointed_along_the_velocity():
    verts, faces, colours = ball.ball_mesh(np.array([10.0, 2.0, 1.5]), np.array([0.0, 0.3, 0.0]))
    assert verts.shape[1] == 3 and faces.shape[1] == 3 and colours.shape == (len(verts), 3)
    c = verts.mean(0)
    assert np.allclose(c, [10.0, 2.0, 1.5], atol=0.02)
    ext = verts.max(0) - verts.min(0)
    assert abs(ext[1] - ball.LENGTH_M) < 0.01 and abs(ext[0] - ball.WIDTH_M) < 0.01 and abs(ext[2] - ball.WIDTH_M) < 0.01
    assert (colours == ball.LACE).all(1).sum() > 0 and (colours == ball.BROWN).all(1).sum() > len(verts) // 2
    still, _, _ = ball.ball_mesh(np.zeros(3), np.zeros(3))                  # no velocity: long axis along +x
    ext2 = still.max(0) - still.min(0)
    assert abs(ext2[0] - ball.LENGTH_M) < 0.01
