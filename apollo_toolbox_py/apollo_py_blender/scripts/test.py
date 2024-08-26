from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V
from apollo_toolbox_py.prelude import *

__all__ = ['tester']


def tester():
    r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
    s = r.get_subdirectory('ur5')
    c = s.to_chain_numpy()
    ch = ChainBlender.spawn(c, r)
    ch.pose_chain([1.,0.,0.,0.,0.,0.])
    print(ch)
    return ch


if __name__ == '__main__':
    tester()
