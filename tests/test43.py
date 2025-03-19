import math
import numpy as np
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import UnitQuaternion
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.rotation_matrices import Rotation3

r = Rotation3.from_axis_angle(V3([1, 1, 1]), 0.1)
print(r)

r = Rotation3.from_axis_angle(V3([-1, -1, -1]), -0.1)
print(r)

# r = Rotation3.from_axis_angle(V3([1, 1, 1]), theta2)
# print(np.linalg.trace(r.array))
# print(r.to_unit_quaternion())

# r = Rotation3.from_axis_angle(V3([1, 2, 3]), math.pi+0.5)
# print(r.to_unit_quaternion())
#
# r = Rotation3.from_axis_angle(V3([1, 2, 3]), math.pi-0.5)
# print(r.to_unit_quaternion())

q = UnitQuaternion.new_random_with_range()
print(q.to_lie_group_h1().ln().vee().norm())

qn = -q
print(qn.to_lie_group_h1().ln().vee().norm())
