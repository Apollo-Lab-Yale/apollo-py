from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V3, V6
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.lie.h1 import LieAlgH1
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.lie.se3_implicit import LieAlgISE3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.lie.se3_implicit_quaternion import LieAlgISE3q
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.lie.so3 import LieAlgSO3, LieGroupSO3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.rotation_matrices import Rotation3
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2, Device, DType

tl.set_backend('pytorch')

v = V6([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
v.array.requires_grad = True
s = LieAlgISE3q.from_euclidean_space_element(v)
print(s)
print(s.exp().ln().vee())

