from apollo_toolbox_py.apollo_py.extra_tensorly_backend import DType, Device
from apollo_toolbox_py.prelude import *

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('ur5')
c = s.to_chain_tensorly()

print(c.urdf_module)
print()

c.to_different_tensorly_backend('pytorch')
print(c.urdf_module)

print()
c.to_different_tensorly_backend('pytorch', device=Device.MPS, dtype=DType.Float32)
print(c.urdf_module)

