import torch

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArray, ApolloPyArrayBackendTorch, \
    ApolloPyArrayBackendJAX

# b = ApolloPyArrayBackendTorch('mps', torch.float32)
b = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_with_backend([1., 2., 3., 4.], b)
c = ApolloPyArray.new_with_backend([1., 2., 3., 4.])
print(a.isclose(c))



