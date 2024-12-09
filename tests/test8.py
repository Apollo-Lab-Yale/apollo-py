from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit import LieGroupISE3

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('ur5')
c = s.to_chain_numpy()
fk_res = c.fk(6*[0.0], LieGroupISE3)

