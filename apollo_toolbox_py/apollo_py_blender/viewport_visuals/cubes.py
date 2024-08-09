import random
from typing import Optional, Union, Tuple, List

import bpy
from easybpy.easybpy import (
    location, rotation, scale_along_local_x, scale_along_local_y, scale_along_local_z, ao,
    create_collection, collection_exists, rename_object, move_object_to_collection,
    select_object, deselect_all_objects, copy_object, delete_object
)
from apollo_toolbox_py.apollo_py_blender.utils.keyframes import KeyframeUtils
from apollo_toolbox_py.apollo_py_blender.utils.material import ApolloBlenderSimpleMaterial
from apollo_toolbox_py.apollo_py_blender.utils.visibility import set_visibility

__all__ = ['ApolloBlenderCube', 'ApolloBlenderCubeSet']


class ApolloBlenderCube:
    """
    Class to create and manage a cube object in Blender.
    """

    def __init__(self) -> None:
        """
        Initialize the Cube object with default values.
        """
        self.object: Optional[bpy.types.Object] = None
        self.center: Optional[Tuple[float, float, float]] = None
        self.local_x_scale: float = 1.0
        self.local_y_scale: float = 1.0
        self.local_z_scale: float = 1.0

    @staticmethod
    def spawn_new(
            center: Union[Tuple[float, float, float], List[float]],
            euler_angles: Union[Tuple[float, float, float], List[float]],
            half_extents: Union[Tuple[float, float, float], List[float]],
            name: Optional[str] = None,
            collection_name: str = 'Cubes',
            material: Optional[ApolloBlenderSimpleMaterial] = None,
            wireframe: bool = False,
            wireframe_thickness: float = 0.01
    ) -> 'ApolloBlenderCube':
        """
        Static method to create a new cube object in Blender.

        Parameters:
        - center: Center point of the cube.
        - euler_angles: Euler angles for the cube's orientation.
        - half_extents: Half extents of the cube along each axis.
        - name: Optional name of the cube object.
        - collection_name: Name of the collection to which the cube belongs.
        - material: Optional material to apply to the cube.
        - wireframe: Whether to add a wireframe modifier to the cube.
        - wireframe_thickness: Thickness of the wireframe.

        Returns:
        - Cube: A new instance of Cube.
        """
        if collection_name is not None:
            exists = collection_exists(collection_name)
            if not exists:
                create_collection(collection_name)

        cube = ApolloBlenderCube()
        bpy.ops.mesh.primitive_cube_add()
        cube.object = ao()
        cube.center = center
        location(cube.object, center)
        rotation(cube.object, euler_angles)
        scale_along_local_x(half_extents[0], cube.object)
        scale_along_local_y(half_extents[1], cube.object)
        scale_along_local_z(half_extents[2], cube.object)
        cube.local_x_scale = half_extents[0]
        cube.local_y_scale = half_extents[1]
        cube.local_z_scale = half_extents[2]

        if name is not None:
            rename_object(cube.object, name)

        if collection_name is not None:
            move_object_to_collection(cube.object, collection_name)

        if wireframe:
            select_object(cube.object)
            bpy.ops.object.modifier_add(type='WIREFRAME')
            cube.object.modifiers['Wireframe'].thickness = wireframe_thickness
            deselect_all_objects()

        if material is not None:
            material.apply_material_to_object(cube.object)

        return cube

    @staticmethod
    def spawn_new_copy(
            cube: 'ApolloBlenderCube',
            center: Union[Tuple[float, float, float], List[float]],
            euler_angles: Union[Tuple[float, float, float], List[float]],
            half_extents: Union[Tuple[float, float, float], List[float]],
            name: Optional[str] = None,
            collection_name: str = 'Cubes',
            material: Optional[ApolloBlenderSimpleMaterial] = None
    ) -> 'ApolloBlenderCube':
        """
        Static method to create a copy of an existing cube object in Blender.

        Parameters:
        - cube: The original cube object to copy.
        - center: Center point of the new cube.
        - euler_angles: Euler angles for the new cube's orientation.
        - half_extents: Half extents of the new cube along each axis.
        - name: Optional name of the new cube object.
        - collection_name: Name of the collection to which the new cube belongs.
        - material: Optional material to apply to the new cube.

        Returns:
        - Cube: A new instance of Cube.
        """
        new_mesh = copy_object(cube.object, collection_name)
        if name is not None:
            rename_object(new_mesh, name)
        out_cube = ApolloBlenderCube()
        out_cube.center = center
        out_cube.object = new_mesh

        out_cube.change_pose(center, euler_angles, half_extents)

        if material is not None:
            material.apply_material_to_object(out_cube.object)

        return out_cube

    def change_pose(
            self,
            center: Union[Tuple[float, float, float], List[float]],
            euler_angles: Union[Tuple[float, float, float], List[float]],
            half_extents: Union[Tuple[float, float, float], List[float]]
    ) -> None:
        """
        Change the position, orientation, and scale of the cube object.

        Parameters:
        - center: New center point of the cube.
        - euler_angles: New Euler angles for the cube's orientation.
        - half_extents: New half extents of the cube along each axis.
        """
        self.center = center
        location(self.object, center)
        rotation(self.object, euler_angles)

        rand_x = random.uniform(0.002, 0.005)
        scale_along_local_x((half_extents[0] + rand_x) / self.local_x_scale, self.object)
        self.local_x_scale = (half_extents[0] + rand_x)

        rand_y = random.uniform(0.002, 0.005)
        scale_along_local_y((half_extents[1] + rand_y) / self.local_y_scale, self.object)
        self.local_y_scale = (half_extents[1] + rand_y)

        rand_z = random.uniform(0.002, 0.005)
        scale_along_local_z((half_extents[2] + rand_z) / self.local_z_scale, self.object)
        self.local_z_scale = (half_extents[2] + rand_z)


class ApolloBlenderCubeSet:
    """
    Class to manage a set of cube objects in Blender.
    """

    def __init__(
            self,
            num_cubes: int,
            collection_name: str = 'CubeSet',
            material_type: str = 'Principled BSDF',
            default_color: Optional[Tuple[float, float, float, float]] = None,
            linked_material_for_each_line: bool = True,
            wireframe: bool = False,
            wireframe_thickness: float = 0.01
    ) -> None:
        """
        Initialize the CubeSet object with a specified number of cubes.

        Parameters:
        - num_cubes: Number of cubes to create in the set.
        - collection_name: Name of the collection to which the cubes belong.
        - material_type: Type of material to apply to the cubes.
        - default_color: Default color of the cubes.
        - linked_material_for_each_line: Whether each cube has a linked material or not.
        - wireframe: Whether to add a wireframe modifier to the cubes.
        - wireframe_thickness: Thickness of the wireframe.
        """
        self.cubes: List[ApolloBlenderCube] = []
        self.materials: List[ApolloBlenderSimpleMaterial] = []

        cube_to_copy: ApolloBlenderCube = ApolloBlenderCube.spawn_new([0, 0, 0], [0, 0, 0], [1, 1, 1],
                                                                      collection_name=None, wireframe=wireframe,
                                                                      wireframe_thickness=wireframe_thickness)

        if default_color is None:
            default_color = (0.2, 0.2, 0.2, 1)

        base_material: ApolloBlenderSimpleMaterial = ApolloBlenderSimpleMaterial(material_type=material_type,
                                                                                 default_color=default_color)
        base_material.keyframe_material(0)

        for i in range(num_cubes):
            cube_copy: ApolloBlenderCube = ApolloBlenderCube.spawn_new_copy(cube_to_copy, [0, 0, 0], [0, 0, 0],
                                                                            [1, 1, 1], collection_name=collection_name)
            if linked_material_for_each_line:
                base_material.apply_material_to_object(cube_copy.object)
                self.materials.append(base_material)
            else:
                new_material: ApolloBlenderSimpleMaterial = ApolloBlenderSimpleMaterial(material_type=material_type,
                                                                                        default_color=default_color)
                new_material.apply_material_to_object(cube_copy.object)
                new_material.keyframe_material(0)
                self.materials.append(new_material)

            set_visibility(cube_copy.object, False)
            KeyframeUtils.keyframe_visibility(cube_copy.object, 0)
            self.cubes.append(cube_copy)

        delete_object(cube_to_copy.object)

        self.per_frame_next_available_cube: List[int] = []

    def set_cube_at_frame(
            self,
            center: Union[Tuple[float, float, float], List[float]],
            euler_angles: Union[Tuple[float, float, float], List[float]],
            half_extents: Union[Tuple[float, float, float], List[float]],
            frame: int,
            color: Optional[Tuple[float, float, float, float]] = None,
            alpha: Optional[float] = None
    ) -> None:
        """
        Set the position, orientation, and visibility of a cube at a specific frame.

        Parameters:
        - center: Center point of the cube.
        - euler_angles: Euler angles for the cube's orientation.
        - half_extents: Half extents of the cube along each axis.
        - frame: Frame number at which the cube is set.
        - color: Color of the cube.
        - alpha: Alpha transparency of the cube.
        """
        while len(self.per_frame_next_available_cube) <= frame:
            self.per_frame_next_available_cube.append(0)

        if len(self.cubes) <= self.per_frame_next_available_cube[frame]:
            print(f'WARNING: Not enough cubes at frame {frame} to draw the given cube.')
            return

        curr_cube: ApolloBlenderCube = self.cubes[self.per_frame_next_available_cube[frame]]
        curr_cube.change_pose(center, euler_angles, half_extents)
        set_visibility(curr_cube.object, True)

        KeyframeUtils.keyframe_transform(curr_cube.object, frame)
        KeyframeUtils.keyframe_visibility(curr_cube.object, frame)

        set_visibility(curr_cube.object, False)
        KeyframeUtils.keyframe_visibility(curr_cube.object, frame + 1)

        if color is not None:
            material: ApolloBlenderSimpleMaterial = self.materials[self.per_frame_next_available_cube[frame]]
            material.set_color(color)
            material.keyframe_material(frame)

        if alpha is not None:
            material: ApolloBlenderSimpleMaterial = self.materials[self.per_frame_next_available_cube[frame]]
            material.set_alpha(alpha)
            material.keyframe_material(frame)

        self.per_frame_next_available_cube[frame] += 1