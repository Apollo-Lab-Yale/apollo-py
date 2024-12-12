import torch

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArrayBackendTorch, ApolloPyArrayBackendJAX, \
    ApolloPyArrayBackendNumpy
from apollo_toolbox_py.apollo_py.apollo_py_linalg.vectors import V3
from apollo_toolbox_py.apollo_py.apollo_py_spatial.rotation_matrices import Rotation3

# backend = ApolloPyArrayBackendTorch(device='mps', dtype=torch.float32)
# backend = ApolloPyArrayBackendJAX()
backend = ApolloPyArrayBackendNumpy()
v = V3([1., 2., 3.], backend)
# v.array.set_torch_requires_grad(True)
r = Rotation3.from_euler_angles(v)
print(r)
