import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArrayBackendNumpy, ApolloPyArrayBackendTorch
from apollo_toolbox_py.apollo_py.apollo_py_linalg.matrices import M, M3
from apollo_toolbox_py.apollo_py.apollo_py_linalg.vectors import V3

backend = ApolloPyArrayBackendTorch()
# backend = ApolloPyArrayBackendNumpy()
v1 = V3([1., 2., 3.], backend)
v2 = V3([1., 2., 4.], backend)

res = v1.cross(v2)
print(res)
print(res.dot(v1))

m1 = M3(np.random.uniform(-1, 1, (3, 3)), backend)
print(m1.rank())

