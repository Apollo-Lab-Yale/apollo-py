import numpy as np
import jax.numpy as jnp
import torch
import math

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArray, ApolloPyArrayBackendTorch, \
    ApolloPyArrayBackendJAX

a = jnp.array([[1.0, 2., 3.0]])
res = a@a.T
print(type(res))
print(res.shape)

print(type(res.item()))

a = torch.tensor([[1., 2., 3.]])
res = a@a.T
print(type(res))
print(res.shape)
print(type(res.item()))

b = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_with_backend([[1., 2., 3.]], b)
res = a@a.T
print(type(res))
print(res.shape)
print(type(res.item()))
