import torch

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArray, ApolloPyArrayBackendJAX, \
    ApolloPyArrayBackendTorch, ApolloPyArrayBackendNumpy, ApolloPyArrayTorch
import numpy as np
import jax.numpy as jnp

backend = ApolloPyArrayBackendTorch('mps', dtype=torch.float32)
# backend = ApolloPyArrayBackendNumpy()
# backend = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_with_backend(np.random.uniform(-3, 3, (3, 3)), backend)
# b = ApolloPyArray.new_with_backend(np.random.uniform(-3, 3, (3, 3)), backend)
# c = ApolloPyArray.new_with_backend(2.0, backend)

a.set_torch_requires_grad(True)
print(a)
d = ApolloPyArray.new_with_backend([[ApolloPyArrayTorch(torch.tensor(1.1)), np.array([2.0]), 3.0], [4, 5, a[0, 0]]], backend)
print(d)




