from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_functions.robot_kinematics_functions import \
    RobotKinematicFunctions
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3, V6
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit import LieGroupISE3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit_quaternion import LieGroupISE3q

res = RobotKinematicFunctions.get_joint_variable_transform('Floating', V3([1, 0, 0]), [1.0, 2., 3., 4, 5, 6], LieGroupISE3, V3, V6)
print(res)


