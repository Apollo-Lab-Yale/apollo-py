
import apollo_rust_file_pyo3 as a

from apollo_py.apollo_py.apollo_py_robotics.robot_directories import RobotPreprocessorRobotsDirectory

fp = a.PathBufPy.new_from_default_apollo_robots_dir()
r = RobotPreprocessorRobotsDirectory(fp)
s = r.get_robot_subdirectory('ur5')

u = s.to_urdf_numpy_module()
print(u.joints[0].origin.pose_m)