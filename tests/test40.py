import numpy as np

from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import UnitQuaternion


# maps 5-vector rep to wxyz quaternion to 5-vector rep
def f(arr):
    v = np.array([arr[0], arr[1], arr[2]])
    v = v / np.linalg.norm(v)
    c = arr[3]
    s = arr[4]
    a = np.array([c, np.abs(s)])
    a = a / np.linalg.norm(a)
    w = a[0]
    s = a[1]
    v = s*v
    out = np.zeros((4, ))
    out[0] = w
    out[1:4] = v
    return out


# maps 4-vector wxyz quaternion to 5-vector rep
def g(arr):
    v = np.array([arr[1], arr[2], arr[3]])
    n = np.linalg.norm(v)
    vn = v / n
    out = np.zeros((5,))
    out[0:3] = vn
    out[3] = arr[0]
    out[4] = n
    return out


max_dis = -1000.0
max_curr_rep = None
max_prev_rep = None

for j in range(10000):
    q = UnitQuaternion.new_random_with_range()
    curr_rep = g(q.array)
    prev_rep = g(q.array)
    for i in range(100):
        print(j, i, max_dis)
        qv = UnitQuaternion.new_random_with_range(-0.001, 0.001)
        q = q*qv
        curr_rep = g(q.array)
        dis = np.linalg.norm(prev_rep - curr_rep)
        if dis > max_dis:
            max_dis = dis
            max_curr_rep = curr_rep
            max_prev_rep = prev_rep
        prev_rep = curr_rep

print(max_dis)
print(max_curr_rep)
print(max_prev_rep)
