from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArray, ApolloPyArrayBackendJAX
import numpy as np

# backend = ApolloPyArrayBackendTorch('mps', dtype=torch.float32)
# backend = ApolloPyArrayBackendNumpy()
backend = ApolloPyArrayBackendJAX()
a = ApolloPyArray.new_with_backend(np.random.uniform(-3, 3, (3, )), backend)
b = ApolloPyArray.new_with_backend(np.random.uniform(-3, 3, (3, )), backend)
c = ApolloPyArray.new_with_backend(2.0, backend)

res = a.cross(b)
print(res)
print(a.dot(res))
print(b.dot(res))


