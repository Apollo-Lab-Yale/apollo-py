import torch
import math

from apollo_toolbox_py.apollo_py.apollo_py_array.apollo_py_array import ApolloPyArrayNumpy, ApolloPyArrayBackendNumpy, \
    ApolloPyArray, ApolloPyArrayBackendJAX, ApolloPyArrayBackendTorch
import numpy as np
import jax.numpy as jnp

# backend = ApolloPyArrayBackendTorch('cpu', dtype=torch.float32)
# backend = ApolloPyArrayBackendNumpy()
backend = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_from_values(np.random.uniform(-1, 1, (4, 4)), backend)
b = ApolloPyArray.new_from_values(np.random.uniform(-1, 1, (4, 4)), backend)

res = a.svd()
print(res.U)
print(res.singular_vals)
print(res.VT)
print(res.U @ res.VT)


