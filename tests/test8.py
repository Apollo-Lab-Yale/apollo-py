from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('ur5')
c = s.to_chain_numpy()
print(c)
