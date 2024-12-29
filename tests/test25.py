from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.quaternions import Quaternion, UnitQuaternion

q = Quaternion([1., 2., 3., 4.])
print(q)
uq = q.to_unit_quaternion()
print(uq)
uuq = UnitQuaternion(q)
print(uuq)
