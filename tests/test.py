import apollo_rust_file_pyo3 as a

from apollo_py.apollo_py.apollo_py_robotics.robot_directories import RobotPreprocessorRobotsDirectory
from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.matrices import M3
from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.rotation_matrices import Rotation3
from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.vectors import V3

fp = a.PathBufPy.new_from_default_apollo_robots_dir()
r = RobotPreprocessorRobotsDirectory(fp)
s = r.get_robot_subdirectory('ur5')

urdf_module = s.to_dof_module()
print(urdf_module)

v3 = V3([1., 2., 3.])
print(v3)

m3 = M3([[1,2,3], [3,4,5], [5,6,7]])
print(m3)

r3 = Rotation3.from_euler_angles([1,2,8])
print(r3.inverse())
