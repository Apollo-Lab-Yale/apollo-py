from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_robotics.robot_runtime_modules.urdf_tensorly_module import \
    ApolloURDFTensorlyModule
import tensorly as tl

tl.set_backend('pytorch')

device = Device.MPS
dtype = DType.Float32

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('ur5')
c = s.to_chain_tensorly(device, dtype)
state = V(6*[0.0], device, dtype)
state.array.requires_grad = True
res = c.fk(state)
print(res)
