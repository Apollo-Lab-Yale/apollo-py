from typing import List

import bpy
from easybpy.easybpy import collection_exists, create_collection, rename_object, ao, location, rotation, \
    move_object_to_collection, set_parent

from apollo_toolbox_py.apollo_py.apollo_py_robotics.resources_directories import ResourcesSubDirectory, \
    ResourcesRootDirectory

__all__ = ['ChainBlender']

from apollo_toolbox_py.apollo_py_blender.utils.mesh_loading import BlenderMeshLoader
from apollo_toolbox_py.apollo_py_blender.utils.transforms import BlenderTransformUtils

from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_robotics.chain_numpy import ChainNumpy
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.se3_implicit_quaternion import LieGroupISE3q


class ChainBlender:
    @classmethod
    def spawn(cls, chain: ChainNumpy, r: ResourcesRootDirectory) -> 'ChainBlender':
        out = cls()

        out.chain = chain

        zeros_state = V(out.chain.dof_module.num_dofs * [0.0])
        fk_res: List[LieGroupISE3q] = out.chain.fk(zeros_state)

        chain_index = ChainBlender.find_next_available_chain_index()
        collection_name = 'Chain_' + str(chain_index)
        create_collection(collection_name)

        for link_idx, frame in enumerate(fk_res):
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            empty_object = ao()
            empty_object.empty_display_size = 0.2
            signature = ChainBlender.get_link_signature(chain_index, link_idx)
            rename_object(ao(), signature)
            position = frame.translation
            euler_angles = frame.rotation.to_rotation_matrix().to_euler_angles()
            location(empty_object, position.array)
            rotation(empty_object, euler_angles)
            move_object_to_collection(empty_object, collection_name)

        links_in_chain = chain.chain_module.links_in_chain
        for link_in_chain in links_in_chain:
            link_idx = link_in_chain.link_idx
            parent_idx = link_in_chain.parent_link_idx
            if parent_idx is not None:
                child = ChainBlender.get_link_signature(chain_index, link_idx)
                parent = ChainBlender.get_link_signature(chain_index, parent_idx)
                set_parent(child, parent)

        blender_objects_plain_meshes_glb = []
        blender_objects_plain_meshes_obj = []
        blender_objects_convex_decomposition_glb = []
        blender_objects_convex_decomposition_obj = []

        ChainBlender._spawn_link_meshes_options(chain, chain_index, chain.plain_meshes_module.recover_full_glb_path_bufs(r), collection_name, 'plain_meshes_glb', blender_objects_plain_meshes_glb)
        ChainBlender._spawn_link_meshes_options(chain, chain_index, chain.plain_meshes_module.recover_full_obj_path_bufs(r), collection_name, 'plain_meshes_obj', blender_objects_plain_meshes_obj)
        ChainBlender._spawn_link_meshes_lists(chain, chain_index, chain.convex_decomposition_meshes_module.recover_full_glb_path_bufs(r), collection_name, 'convex_decomposition_meshes_glb', blender_objects_convex_decomposition_glb)
        ChainBlender._spawn_link_meshes_lists(chain, chain_index, chain.convex_decomposition_meshes_module.recover_full_obj_path_bufs(r), collection_name, 'convex_decomposition_meshes_obj', blender_objects_convex_decomposition_obj)

        return out

    @staticmethod
    def _spawn_link_meshes_options(chain, chain_index, file_paths, collection_name, suffix, blender_objects_list):
        for link_idx, path in enumerate(file_paths):
            tmp = []
            if path is not None:
                link_name = chain.urdf_module.links[link_idx].name
                mesh_name = link_name + '_' + suffix
                blender_object = BlenderMeshLoader.import_mesh_file(mesh_name, path.to_string(), collection_name)
                tmp.append(blender_object)
                parent_name = ChainBlender.get_link_signature(chain_index, link_idx)
                BlenderTransformUtils.copy_location_and_rotation(parent_name, blender_object)
                set_parent(blender_object, parent_name)
                move_object_to_collection(blender_object, collection_name)

            blender_objects_list.append(tmp)

    @staticmethod
    def _spawn_link_meshes_lists(chain, chain_index, file_paths, collection_name, suffix, blender_objects_list):
        for link_idx, file_path_list in enumerate(file_paths):
            tmp = []
            for subcomponent_idx, path in enumerate(file_path_list):
                link_name = chain.urdf_module.links[link_idx].name
                mesh_name = link_name + '_' + str(subcomponent_idx) + '_' + suffix
                blender_object = BlenderMeshLoader.import_mesh_file(mesh_name, path.to_string(), collection_name)
                tmp.append(blender_object)
                parent_name = ChainBlender.get_link_signature(chain_index, link_idx)
                BlenderTransformUtils.copy_location_and_rotation(parent_name, blender_object)
                set_parent(blender_object, parent_name)
                move_object_to_collection(blender_object, collection_name)
            blender_objects_list.append(tmp)

    @staticmethod
    def find_next_available_chain_index() -> int:
        chain_index = 0
        while True:
            collection_name = 'Chain_' + str(chain_index)
            if collection_exists(collection_name):
                chain_index += 1
            else:
                break
        return chain_index

    @staticmethod
    def get_link_signature(chain_index: int, link_idx: int):
        return 'chain_' + str(chain_index) + '_link_' + str(link_idx)
