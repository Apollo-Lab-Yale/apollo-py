import numpy as np

from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M, M3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V, V3
import tensorly as tl

from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.rotation_matrices import Rotation3

tl.set_backend('numpy')

v = V3([1., 2., 3.], Device.CPU, DType.Float64)
v2 = V3([0.2, -0.4, 0.7], Device.CPU, DType.Float64)
# v.array.requires_grad = True
r = Rotation3.from_look_at(v, v2)
print(r.map_point(v))


