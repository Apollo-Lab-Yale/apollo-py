from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V
from apollo_toolbox_py.prelude import *

__all__ = ['tester']


def tester():
    r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
    s = r.get_subdirectory('ur5')
    c = s.to_chain_numpy()
    ch = ChainBlender.spawn(c, r)
    ch.set_state([1., 0., 0., 0., 0., 0.])
    ch.keyframe_state(1)
    ch.set_state([1., 1., 0., 0., 0., 0.])
    ch.keyframe_state(10)
    ch.set_link_plain_mesh_color(1, (1., 0., 0., 1.0))
    ch.set_link_plain_mesh_alpha(1, 0.1)
    ch.keyframe_plain_mesh_material(1, 1)
    ch.set_link_plain_mesh_color(1, (0., 1., 0., 1.0))
    ch.set_link_plain_mesh_alpha(1, 0.5)
    ch.keyframe_plain_mesh_material(10, 1)
    print(ch)
    return ch


if __name__ == '__main__':
    tester()
