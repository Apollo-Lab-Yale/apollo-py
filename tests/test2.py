from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory
from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_preprocessed_modules.mesh_modules.utils import \
    recover_full_mesh_path_bufs_from_relative_mesh_paths_options
from apollo_toolbox_py.apollo_py.path_buf import PathBufPyWrapper
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3, V
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_robotics.chain_numpy import ChainNumpy
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.isometries import Isometry3, IsometryMatrix3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit import LieGroupISE3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit_quaternion import LieGroupISE3q
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import UnitQuaternion

p = PathBufPyWrapper.new_from_default_apollo_robots_dir()
r = ResourcesRootDirectory(p)
s = r.get_subdirectory('ur5')
chain = ChainNumpy(s)

res = chain.fk(V([1., 2., 3., 4., 5., 6.]))
print(res)
