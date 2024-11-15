import apollo_rust_file_pyo3 as a

from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesRootDirectory
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.isometries import Isometry3, IsometryMatrix3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit import LieGroupISE3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit_quaternion import LieGroupISE3q
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.matrices import M3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import Quaternion, UnitQuaternion
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.rotation_matrices import Rotation3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3

fp = a.PathBufPy.new_from_default_apollo_robots_dir()
r = ResourcesRootDirectory(fp)
s = r.get_subdirectory('ur5')

urdf_module = s.to_dof_module()
print(urdf_module)

v3 = V3([1., 2., 3.])
print(v3)

m3 = M3([[1,2,3], [3,4,5], [5,6,7]])
print(m3)

r3 = Rotation3.from_euler_angles([1,2,8])
print(r3.inverse())

q = Quaternion([1,2,3,4])
q2 = Quaternion([2,3,4,5])
print(q@q2)

q = UnitQuaternion([1,0,0,0])
print(q)
q3 = UnitQuaternion([1,0,0,0])
print(q@q3)

i = Isometry3(q3, V3([1,2,3]))
print(i)

i2 = IsometryMatrix3(r3, V3([1,2,3]))
i3 = IsometryMatrix3(r3, V3([1,2,3]))
print(i2@i3)

h = q3.to_lie_group_h1()
print(h.ln().exp())

s = r3.to_lie_group_so3()
print(s.ln())

ll = LieGroupISE3q(UnitQuaternion.new_unchecked([1,0,0,0]), V3([1,2,3]))
print(ll.ln().exp())

ll2 = LieGroupISE3(Rotation3.from_euler_angles([1,2,3]), V3([1,2,3]))
print(ll2)
print(ll2.ln().exp())

