from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_robotics.robot_runtime_modules.urdf_tensorly_module import \
    ApolloURDFTensorlyModule
import tensorly as tl

tl.set_backend('pytorch')

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('ur5')
c = s.to_chain()
u = c.base_urdf_module

uu = ApolloURDFTensorlyModule.from_urdf_module(u, Device.MPS, DType.Float32)
print(uu)