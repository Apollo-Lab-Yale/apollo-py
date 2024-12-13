import numpy as np
import torch

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArray, ApolloPyArrayBackendTorch, \
    ApolloPyArrayBackendJAX

# b = ApolloPyArrayBackendTorch(device='mps', dtype=torch.float32)
b = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_with_backend([[1., 2., 3.], [4, 5, 6]], b)
print(a)
a = a + 8.0
print(a)




