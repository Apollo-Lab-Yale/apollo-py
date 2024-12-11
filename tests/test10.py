from apollo_toolbox_py.apollo_py.apollo_py_array.apollo_py_array import ApolloPyArrayBackendTorch, \
    ApolloPyArrayBackendJAX
from apollo_toolbox_py.apollo_py.apollo_py_linalg.vectors import V

backend = ApolloPyArrayBackendJAX()
v1 = V([1., 2., 3.], backend)
v2 = V([1., 2., 3.], backend)

res = v1.unit()
print(res.norm())

