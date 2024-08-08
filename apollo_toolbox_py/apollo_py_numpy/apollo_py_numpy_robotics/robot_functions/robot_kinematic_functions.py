from typing import List

from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_preprocessed_modules.chain_module import ApolloChainModule
from apollo_toolbox_py.apollo_py.apollo_py_robotics.robot_preprocessed_modules.dof_module import ApolloDOFModule
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3, V6, V
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_robotics.robot_runtime_modules.urdf_numpy_module import \
    ApolloURDFNumpyModule
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit_quaternion import LieGroupISE3q, \
    LieAlgISE3q
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import UnitQuaternion


class RobotKinematicFunctions:
    @staticmethod
    def fk(state: V, urdf_module: ApolloURDFNumpyModule, chain_module: ApolloChainModule,
           dof_module: ApolloDOFModule) -> List[LieGroupISE3q]:
        links = urdf_module.links
        joints = urdf_module.joints
        kinematic_hierarchy = chain_module.kinematic_hierarchy
        joint_idx_to_dofs_mapping = dof_module.joint_idx_to_dof_idxs_mapping

        num_links = len(links)
        out = [LieGroupISE3q.identity() for _ in range(num_links)]

        for i, layer in enumerate(kinematic_hierarchy):
            if i == 0:
                continue

            for link_idx in layer:
                link_in_chain = chain_module.links_in_chain[link_idx]
                parent_link_idx = link_in_chain.parent_link_idx
                parent_joint_idx = link_in_chain.parent_joint_idx
                parent_joint = joints[parent_joint_idx]

                constant_transform = parent_joint.origin.pose
                dof_idxs = joint_idx_to_dofs_mapping[parent_joint_idx]
                joint_dofs = [state[i] for i in dof_idxs]
                joint_axis = parent_joint.axis.xyz
                joint_type = parent_joint.joint_type
                variable_transform = RobotKinematicFunctions.get_joint_variable_transform(joint_type, joint_axis,
                                                                                          joint_dofs)
                out[link_idx] = out[parent_link_idx].group_operator(constant_transform).group_operator(
                    variable_transform)

        return out

    @staticmethod
    def get_joint_variable_transform(joint_type: str, joint_axis: V3, joint_dofs: List[float]) -> LieGroupISE3q:
        if joint_type == 'Revolute':
            assert len(joint_dofs) == 1
            sa: V3 = joint_dofs[0] * joint_axis
            return LieGroupISE3q(UnitQuaternion.from_scaled_axis(sa), V3([0, 0, 0]))
        elif joint_type == 'Continuous':
            assert len(joint_dofs) == 1
            sa: V3 = joint_dofs[0] * joint_axis
            return LieGroupISE3q(UnitQuaternion.from_scaled_axis(sa), V3([0, 0, 0]))
        elif joint_type == 'Prismatic':
            assert len(joint_dofs) == 1
            sa: V3 = joint_dofs[0] * joint_axis
            return LieGroupISE3q(UnitQuaternion.new_unchecked([1, 0, 0, 0]), sa)
        elif joint_type == 'Fixed':
            assert len(joint_dofs) == 0
            return LieGroupISE3q.identity()
        elif joint_type == 'Floating':
            assert len(joint_dofs) == 0
            v6 = V6(joint_dofs)
            return LieAlgISE3q.from_euclidean_space_element(v6).exp()
        elif joint_type == 'Planar':
            assert len(joint_dofs) == 2
            t = V3([joint_dofs[0], joint_dofs[1], 0.0])
            return LieGroupISE3q(UnitQuaternion.new_unchecked([1, 0, 0, 0]), t)
        elif joint_type == 'Spherical':
            assert len(joint_dofs) == 3
            v = V3(joint_dofs)
            return LieGroupISE3q(UnitQuaternion.from_scaled_axis(v), V3([0, 0, 0]))
        else:
            raise ValueError(f"not valid joint type: {joint_type}")
