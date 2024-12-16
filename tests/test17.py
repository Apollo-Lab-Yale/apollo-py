import numpy as np

from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M, M3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V, V3
import tensorly as tl

from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.isometries import Isometry3, IsometryMatrix3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.quaternions import Quaternion, UnitQuaternion
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.rotation_matrices import Rotation3

tl.set_backend('jax')

v1 = V3([1., 2., 3.], Device.MPS, DType.Float32)
v2 = V3([1., 2., 3.], Device.MPS, DType.Float32)
v3 = V3([1., 2., 3.], Device.MPS, DType.Float32)

v1.array.requires_grad = True
v2.array.requires_grad = True

i = IsometryMatrix3.from_scaled_axis(v1, v2)
print(i)

res = i.map_point(v3)
print(res)

