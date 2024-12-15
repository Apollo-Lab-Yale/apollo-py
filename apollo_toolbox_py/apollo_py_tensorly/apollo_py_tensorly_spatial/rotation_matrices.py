from typing import Union, List

import numpy as np
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2, Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M3
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V3


class Rotation3(M3):
    def __init__(self, array: Union[List[List[float]], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        super().__init__(array, device, dtype)
        if not T2.allclose(self.array @ self.array.T, np.eye(3), rtol=1e-5, atol=1e-5):
            raise ValueError("Rotation matrix must be orthonormal.")

    @classmethod
    def new_unchecked(cls, array: Union[List[List[float]], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64) -> 'Rotation3':
        out = cls.__new__(cls)
        m = M3(array, device, dtype)
        out.array = m.array
        return out

    @classmethod
    def new_normalize(cls, array: Union[List[List[float]], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64) -> 'Rotation3':
        m = M3(array, device, dtype)

        u, _, vh = m.svd(True)
        array = u.array @ vh.array

        if T2.det(array) < 0:
            # u[:, -1] *= -1
            u = u.set_and_return((slice(None), 0), -1.0*u[:, 0])
            array = u.array @ vh.array

        return cls(array)

    @classmethod
    def from_euler_angles(cls, xyz: V3) -> 'Rotation3':
        roll, pitch, yaw = xyz.array

        R_x = tl.zeros((3, 3), device=xyz.array.device, dtype=xyz.array.dtype)
        R_x = T2.set_and_return(R_x, (0, 0), 1.0)
        R_x = T2.set_and_return(R_x, (1, 1), tl.cos(roll))
        R_x = T2.set_and_return(R_x, (1, 2), -tl.sin(roll))
        R_x = T2.set_and_return(R_x, (2, 1), tl.sin(roll))
        R_x = T2.set_and_return(R_x, (2, 2), tl.cos(roll))

        R_y = tl.zeros((3, 3), device=xyz.array.device, dtype=xyz.array.dtype)
        R_y = T2.set_and_return(R_y, (1, 1), 1.0)
        R_y = T2.set_and_return(R_y, (0, 0), tl.cos(pitch))
        R_y = T2.set_and_return(R_y, (0, 2), tl.sin(pitch))
        R_y = T2.set_and_return(R_y, (2, 0), -tl.sin(pitch))
        R_y = T2.set_and_return(R_y, (2, 2), tl.cos(pitch))

        R_z = tl.zeros((3, 3), device=xyz.array.device, dtype=xyz.array.dtype)
        R_z = T2.set_and_return(R_z, (2, 2), 1.0)
        R_z = T2.set_and_return(R_z, (0, 0), tl.cos(yaw))
        R_z = T2.set_and_return(R_z, (0, 1), -tl.sin(yaw))
        R_z = T2.set_and_return(R_z, (1, 0), tl.sin(yaw))
        R_z = T2.set_and_return(R_z, (1, 1), tl.cos(yaw))

        rotation_matrix = R_z @ R_y @ R_x

        return cls(rotation_matrix)

    @classmethod
    def from_axis_angle(cls, axis: V3, angle) -> 'Rotation3':
        if angle == 0.0:
            return Rotation3.new_unchecked(tl.eye(3))

        axis = axis.normalize()
        x, y, z = axis.array
        cos_theta = tl.cos(tl.tensor(angle, device=axis.array.device, dtype=axis.array.dtype))
        sin_theta = tl.sin(tl.tensor(angle, device=axis.array.device, dtype=axis.array.dtype))
        one_minus_cos = 1.0 - cos_theta

        rotation_matrix = tl.zeros((3, 3), device=axis.array.device, dtype=axis.array.dtype)
        rotation_matrix = T2.set_and_return(rotation_matrix, (0, 0), cos_theta + x * x * one_minus_cos)
        rotation_matrix = T2.set_and_return(rotation_matrix, (0, 1), x * y * one_minus_cos - z * sin_theta)
        rotation_matrix = T2.set_and_return(rotation_matrix, (0, 2), x * z * one_minus_cos + y * sin_theta)
        rotation_matrix = T2.set_and_return(rotation_matrix, (1, 0), y * x * one_minus_cos + z * sin_theta)
        rotation_matrix = T2.set_and_return(rotation_matrix, (1, 1), cos_theta + y * y * one_minus_cos)
        rotation_matrix = T2.set_and_return(rotation_matrix, (1, 2), y * z * one_minus_cos - x * sin_theta)
        rotation_matrix = T2.set_and_return(rotation_matrix, (2, 0), z * x * one_minus_cos - y * sin_theta)
        rotation_matrix = T2.set_and_return(rotation_matrix, (2, 1), z * y * one_minus_cos + x * sin_theta)
        rotation_matrix = T2.set_and_return(rotation_matrix, (2, 2), cos_theta + z * z * one_minus_cos)

        return cls(rotation_matrix)

    @classmethod
    def from_scaled_axis(cls, scaled_axis: V3) -> 'Rotation3':
        n = scaled_axis.norm()
        return Rotation3.from_axis_angle(scaled_axis, n)

    @classmethod
    def from_look_at(cls, look_at_vector: V3, axis: V3) -> 'Rotation3':
        """
        rotates identity matrix such that axis will align with look_at_vector
        """
        look_at_vector = look_at_vector.normalize()
        axis = axis.normalize()

        rotation_axis = axis.cross(look_at_vector)
        # angle = np.acos(min(axis.dot(look_at_vector), 1.0))
        angle = tl.acos(axis.dot(look_at_vector))

        return Rotation3.from_axis_angle(rotation_axis, angle)

    def transpose(self) -> 'Rotation3':
        return Rotation3(self.array.T)

    def to_euler_angles(self) -> V3:
        m = self.array
        if m[2][0] < 1:
            if m[2][0] > -1:
                pitch = tl.arcsin(-m[2][0])
                roll = T2.arctan2(m[2][1], m[2][2])
                yaw = T2.arctan2(m[1][0], m[0][0])
            else:
                pitch = tl.pi / 2
                roll = T2.arctan2(-m[1][2], m[1][1])
                yaw = 0
        else:
            pitch = -tl.pi / 2
            roll = T2.arctan2(-m[1][2], m[1][1])
            yaw = 0

        out = V3([0., 0., 0.])
        out[0] = roll
        out[1] = pitch
        out[2] = yaw
        return out

    def inverse(self) -> 'Rotation3':
        return self.new_unchecked(self.array.T)

    def map_point(self, v: V3) -> 'V3':
        return V3(self.array@v.array)

    def __repr__(self) -> str:
        return f"Rotation3(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"Rotation3(\n{self.array}\n)"