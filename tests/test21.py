from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend, Device, DType
from apollo_toolbox_py.prelude import *


r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('ur5')
c = s.to_chain_tensorly(Backend.PyTorch, Device.MPS, DType.Float32)

print(c.urdf_module)

