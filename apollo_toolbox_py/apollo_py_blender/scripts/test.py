from apollo_toolbox_py.prelude import *

__all__ = ['tester']


def tester():
    r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
    s = r.get_subdirectory('ur5')
    c = s.to_chain_numpy()
    ch = ChainBlender.spawn(c, r)
    print(ch)


tester()
