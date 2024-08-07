from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory
from apollo_toolbox_py.apollo_py.path_buf import PathBufPyWrapper

p = PathBufPyWrapper.new_from_default_apollo_robots_dir()
r = ResourcesRootDirectory(p)
s = r.get_subdirectory('ur5')

