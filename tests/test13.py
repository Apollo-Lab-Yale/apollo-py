import torch

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArrayBackendTorch
from apollo_toolbox_py.apollo_py.apollo_py_linalg.vectors import V3
from apollo_toolbox_py.apollo_py.apollo_py_spatial.rotation_matrices import Rotation3

b = ApolloPyArrayBackendTorch(device='cpu', dtype=torch.float32)
v = V3([1., 2., 3.], b)
v.array.set_torch_requires_grad(True)
r = Rotation3.from_axis_angle(v, 1.0)
print(r)

print(r.det())
