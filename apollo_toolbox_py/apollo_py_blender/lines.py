from easybpy.easybpy import *
import numpy as np


class Line:
    def __init__(self):
        self.radius = None
        self.line_mesh = None
        self.name = None
        self.length = 2.0

    @staticmethod
    def spawn_new(start_point, end_point, radius=0.01, vertices=6, name=None, collection_name='Lines', material=None):
        line = Line()
        line.radius = radius
        line.name = name

        if collection_name is not None:
            exists = collection_exists(collection_name)
            if not exists:
                create_collection(collection_name)
        bpy.ops.mesh.primitive_cylinder_add(depth=2, vertices=vertices, scale=(radius, radius, 1))
        object = ao()
        if name is not None:
            rename_object(object, name)

        if collection_name is not None:
            move_object_to_collection(object, collection_name)

        line.line_mesh = object

        # line.change_pose(start_point, end_point)

        if material is not None:
            material.apply_material_to_object(line.line_mesh)

        return line

    @staticmethod
    def spawn_new_copy(line, start_point, end_point, radius=0.01, name=None, collection_name='Lines', material=None):
        new_mesh = copy_object(line.line_mesh, collection_name)
        if name is not None:
            rename_object(new_mesh, name)
        out_line = Line()
        out_line.radius = radius
        out_line.line_mesh = new_mesh
        out_line.name = name
        out_line.length = line.length

        out_line.change_pose(start_point, end_point)

        if material is not None:
            material.apply_material_to_object(out_line.line_mesh)

        return out_line

    def change_pose(self, start_point, end_point):
        s = np.array(start_point)
        e = np.array(end_point)
        d = e - s
        center = (s + e) / 2.0
        length = np.linalg.norm(d)

        r = None
        # r = optima.OptimaRotationPy.new_rotation_matrix_from_lookat_py([d[0], d[1], d[2]], "Z")
        euler_angles = r.to_euler_angles_py()
        if euler_angles == [0., 0., 0.]:
            print('ERROR in line change pose: ', start_point, end_point)

        scale_along_local_z((1.0 / self.length) * length, self.line_mesh)
        location(self.line_mesh, [center[0], center[1], center[2]])
        rotation(self.line_mesh, [euler_angles[0], euler_angles[1], euler_angles[2]])

        self.length = length

    def change_radius(self, radius):
        scale_along_local_x((1.0 / self.radius) * radius, self.line_mesh)
        scale_along_local_y((1.0 / self.radius) * radius, self.line_mesh)

        self.radius = radius
