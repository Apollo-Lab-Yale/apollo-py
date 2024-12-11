import torch
import math

from apollo_toolbox_py.apollo_py.apollo_py_array.apollo_py_array import ApolloPyArrayNumpy, ApolloPyArrayBackendNumpy, \
    ApolloPyArray, ApolloPyArrayBackendJAX, ApolloPyArrayBackendTorch, ApolloPyArrayJAX
import numpy as np
import jax.numpy as jnp

backend = ApolloPyArrayBackendTorch('mps', dtype=torch.float32)
# backend = ApolloPyArrayBackendNumpy()
# backend = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_with_backend(np.random.uniform(-3, 3, (4, 4)), backend)
b = ApolloPyArray.new_with_backend(np.random.uniform(-3, 3, (4, 4)), backend)

print(a)
print(a**3)

print(b)
c = ApolloPyArray.new_with_backend(b)
print(c)

