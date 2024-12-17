from typing import Union, Type, List

from apollo_toolbox_py.apollo_py.apollo_py_robotics.chain import Chain
from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesSubDirectory
from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_functions.robot_kinematics_functions import \
    RobotKinematicFunctions
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V, V3, V6
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.lie.se3_implicit import LieGroupISE3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_spatial.lie.se3_implicit_quaternion import LieGroupISE3q


class ChainTensorly(Chain):
    def __init__(self, s: ResourcesSubDirectory, device: Device = Device.CPU, dtype: DType = DType.Float64):
        super().__init__(s)
        self.device = device
        self.dtype = dtype
        self.urdf_module = s.to_urdf_tensorly_module(device, dtype)

    def fk(self, state: V, lie_group_type: Union[Type[LieGroupISE3q], Type[LieGroupISE3]] = LieGroupISE3q) -> List[
        Union[LieGroupISE3q, LieGroupISE3]]:
        return RobotKinematicFunctions.fk(state, self.urdf_module, self.chain_module, self.dof_module, lie_group_type,
                                          V3, V6, self.device, self.dtype)

    def reverse_of_fk(self, link_frames: List[Union[LieGroupISE3q, LieGroupISE3]],
                      lie_group_type: Union[Type[LieGroupISE3q], Type[LieGroupISE3]] = LieGroupISE3q):
        return RobotKinematicFunctions.reverse_of_fk(link_frames, self.urdf_module, self.chain_module, self.dof_module,
                                                     lie_group_type, V, V3)