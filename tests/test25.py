from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.quaternions import Quaternion, UnitQuaternion

q = Quaternion([1., 2., 3., 4.])
uq = q.to_unit_quaternion()
uuq = UnitQuaternion(q)

q = uuq.to_lie_group_h1()
print(q)

q2 = uuq.to_lie_group_h1()
print(q2)

print(q.displacement(q2).ln().vee())
