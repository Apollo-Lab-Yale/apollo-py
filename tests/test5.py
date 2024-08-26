
from apollo_toolbox_py.prelude import *

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('b1')
c = s.to_chain_numpy()

