from typing import Optional

import bpy
from easybpy.easybpy import rename_object, ao, get_object, so, delete_object, set_parent, move_object_to_collection

__all__ = ['MeshLoader']


class MeshLoader:
    @staticmethod
    def import_stl(object_name: str, filepath: str, collection_name: Optional[str] = None):
        bpy.ops.import_mesh.stl(filepath=filepath)
        rename_object(ao(), object_name)
        if collection_name is not None:
            move_object_to_collection(object_name, collection_name)

    @staticmethod
    def import_obj(object_name: str, filepath: str, collection_name: Optional[str] = None):
        bpy.ops.import_scene.obj(filepath=filepath)
        rename_object(ao(), object_name)
        if collection_name is not None:
            move_object_to_collection(object_name, collection_name)

    @staticmethod
    def import_dae(object_name: str, filepath: str, collection_name: Optional[str] = None):
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        rename_object(ao(), object_name)
        empty_object = get_object(object_name)
        empty_object.empty_display_size = 0.002
        bpy.ops.wm.collada_import(filepath=filepath)
        for s in so():
            if s.type != 'MESH':
                delete_object(s)
            else:
                set_parent(s, empty_object)

            if collection_name is not None:
                move_object_to_collection(s, collection_name)

    @staticmethod
    def import_glb(object_name: str, filepath: str, collection_name: Optional[str] = None):
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        rename_object(ao(), object_name)
        empty_object = get_object(object_name)
        empty_object.empty_display_size = 0.002
        bpy.ops.import_scene.gltf(filepath=filepath)
        for s in so():
            if s.type != 'MESH':
                delete_object(s)
            else:
                set_parent(s, empty_object)

            if collection_name is not None:
                move_object_to_collection(s, collection_name)

    @staticmethod
    def import_mesh_file(object_name, filepath, collection_name=None):
        split = filepath.split('.')
        if len(split) == 0:
            return
        ext = split[-1]
        if ext == 'stl' or ext == 'STL':
            MeshLoader.import_stl(object_name, filepath, collection_name)
        elif ext == 'obj' or ext == 'OBJ':
            MeshLoader.import_obj(object_name, filepath, collection_name)
        elif ext == 'dae' or ext == 'DAE':
            MeshLoader.import_dae(object_name, filepath, collection_name)
        elif ext == 'glb' or ext == 'gltf' or ext == 'GLB' or ext == 'GLTF':
            MeshLoader.import_glb(object_name, filepath, collection_name)
