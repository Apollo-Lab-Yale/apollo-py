import torch

from apollo_toolbox_py.apollo_py.apollo_py_array.apollo_py_array import ApolloPyArrayNumpy, ApolloPyArrayBackendNumpy, \
    ApolloPyArray, ApolloPyArrayBackendJAX, ApolloPyArrayBackendTorch
import numpy as np
import jax.numpy as jnp

# backend = ApolloPyArrayBackendTorch('mps', dtype=torch.float32)
# backend = ApolloPyArrayBackendNumpy()
backend = ApolloPyArrayBackendJAX()
a = ApolloPyArray(np.random.uniform(-1, 1, (4, 4)), backend)
b = ApolloPyArray(np.random.uniform(-1, 1, (4, 4)), backend)

print(a)
print(a[0, 0])
a[:, 0] = b[0, :]
print(b)
print(a + b)

