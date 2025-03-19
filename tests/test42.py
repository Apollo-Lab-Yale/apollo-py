import numpy as np

from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import UnitQuaternion
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.rotation_matrices import Rotation3


max_dis = -1000.0
max_curr_rep = None
max_prev_rep = None

for j in range(10000):
    r = UnitQuaternion.new_random_with_range()
    curr_rep = r.to_rotation_matrix().array.flatten()
    prev_rep = r.to_rotation_matrix().array.flatten()
    for i in range(1000):
        print(j, i, max_dis)
        rv = UnitQuaternion.new_random_with_range(-0.001, 0.001)
        r = r*rv
        curr_rep = r.to_rotation_matrix().array.flatten()
        dis = np.linalg.norm(prev_rep - curr_rep)
        if dis > max_dis:
            max_dis = dis
            max_curr_rep = curr_rep
            max_prev_rep = prev_rep
        prev_rep = curr_rep

print(max_dis)
print(max_curr_rep)
print(max_prev_rep)
