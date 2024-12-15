import numpy as np

from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M, M3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V, V3
import tensorly as tl

from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.quaternions import Quaternion, UnitQuaternion
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.rotation_matrices import Rotation3

tl.set_backend('numpy')

e = V3([1., 2., 3.])
# e.array.requires_grad = True
q = UnitQuaternion.from_scaled_axis(e)
print(q.inverse() @ q)



