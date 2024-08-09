__all__ = ['RobotKinematicFunctions']

from typing import Union, TypeVar, Type, List

from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_preprocessed_modules.chain_module import ApolloChainModule
from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_preprocessed_modules.dof_module import ApolloDOFModule
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3 as V3Numpy, V6 as V6Numpy, V as VNumpy
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_robotics.robot_runtime_modules.urdf_numpy_module import \
    ApolloURDFNumpyModule
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit import LieGroupISE3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit_quaternion import LieGroupISE3q

U = TypeVar('U', bound=Union[ApolloURDFNumpyModule])
LT = TypeVar('LT', bound=Union[Type[LieGroupISE3q], Type[LieGroupISE3]])
L = TypeVar('L', bound=Union[LieGroupISE3q, LieGroupISE3])
V3 = TypeVar('V3', bound=Union[V3Numpy])
V6 = TypeVar('V6', bound=Union[V6Numpy])
V = TypeVar('V', bound=Union[VNumpy])


class RobotKinematicFunctions:
    @staticmethod
    def fk(state: V, urdf_module: U, chain_module: ApolloChainModule, dof_module: ApolloDOFModule,
           lie_group_type: LT, vector3_type: V3, vector6_type: V6) -> List[L]:
        links = urdf_module.links
        joints = urdf_module.joints
        kinematic_hierarchy = chain_module.kinematic_hierarchy
        joint_idx_to_dofs_mapping = dof_module.joint_idx_to_dof_idxs_mapping

        num_links = len(links)
        out = [lie_group_type.identity() for _ in range(num_links)]

        for i, layer in enumerate(kinematic_hierarchy):
            if i == 0:
                continue

            for link_idx in layer:
                link_in_chain = chain_module.links_in_chain[link_idx]
                parent_link_idx = link_in_chain.parent_link_idx
                parent_joint_idx = link_in_chain.parent_joint_idx
                parent_joint = joints[parent_joint_idx]

                constant_transform = parent_joint.origin.get_pose_from_lie_group_type(lie_group_type)
                dof_idxs = joint_idx_to_dofs_mapping[parent_joint_idx]
                joint_dofs = [state[i] for i in dof_idxs]
                joint_axis = parent_joint.axis.xyz
                joint_type = parent_joint.joint_type
                variable_transform = RobotKinematicFunctions.get_joint_variable_transform(joint_type, joint_axis,
                                                                                          joint_dofs, lie_group_type,
                                                                                          vector3_type, vector6_type)
                out[link_idx] = out[parent_link_idx].group_operator(constant_transform).group_operator \
                    (variable_transform)

        return out

    @staticmethod
    def reverse_of_fk(link_frames: List[L], urdf_module: U, chain_module: ApolloChainModule,
                      dof_module: ApolloDOFModule,
                      lie_group_type: LT, vector3_type: V3, vector6_type: V6) -> V:
        out = V(dof_module.num_dofs * [0.0])

        for joint_in_chain in chain_module.joints_in_chain:
            joint_idx = joint_in_chain.joint_idx
            parent_link_idx = joint_in_chain.parent_link_idx
            child_link_idx = joint_in_chain.child_link_idx
            joint = urdf_module.joints[joint_idx]
            constant_transform = joint.origin.get_pose_from_lie_group_type(lie_group_type)
            joint_type = joint.joint_type
            dof_idxs = dof_module.joint_idx_to_dof_idxs_mapping[joint_idx]

        return out

    @staticmethod
    def get_joint_variable_transform(joint_type: str, joint_axis, joint_dofs, lie_group_type: LT, vector3_type: V3,
                                     vector6_type: V6):
        if joint_type == 'Revolute':
            assert len(joint_dofs) == 1
            sa = joint_dofs[0] * joint_axis
            return lie_group_type.from_scaled_axis(sa, vector3_type([0, 0, 0]))
        elif joint_type == 'Continuous':
            assert len(joint_dofs) == 1
            sa = joint_dofs[0] * joint_axis
            return lie_group_type.from_scaled_axis(sa, vector3_type([0, 0, 0]))
        elif joint_type == 'Prismatic':
            assert len(joint_dofs) == 1
            sa = joint_dofs[0] * joint_axis
            return lie_group_type.from_scaled_axis(vector3_type([0, 0, 0]), sa)
        elif joint_type == 'Fixed':
            assert len(joint_dofs) == 0
            return lie_group_type.identity()
        elif joint_type == 'Floating':
            assert len(joint_dofs) == 6
            v6 = vector6_type(joint_dofs)
            return lie_group_type.get_lie_alg_type().from_euclidean_space_element(v6).exp()
        elif joint_type == 'Planar':
            assert len(joint_dofs) == 2
            t = vector3_type([joint_dofs[0], joint_dofs[1], 0.0])
            return lie_group_type.from_scaled_axis(vector3_type([0, 0, 0]), t)
        elif joint_type == 'Spherical':
            assert len(joint_dofs) == 3
            v = vector3_type(joint_dofs)
            return lie_group_type.from_scaled_axis(v, vector3_type([0, 0, 0]))
        else:
            raise ValueError(f"not valid joint type: {joint_type}")
