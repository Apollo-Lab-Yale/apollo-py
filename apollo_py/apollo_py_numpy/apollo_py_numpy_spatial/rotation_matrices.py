from typing import Union, List

import numpy as np

from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.matrices import M3
from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.vectors import V3


class Rotation3(M3):
    def __init__(self, array: Union[List[List[float]], np.ndarray]):
        super().__init__(array)
        if not np.allclose(self.array @ self.array.T, np.eye(3), rtol=1e-7, atol=1e-7):
            raise ValueError("Rotation matrix must be orthonormal.")

    @classmethod
    def new_unchecked(cls, array: Union[List[List[float]], np.ndarray]) -> 'Rotation3':
        out = cls.__new__(cls)
        out.array = array
        return out

    @classmethod
    def new_normalize(cls, array: Union[List[List[float]], np.ndarray]) -> 'Rotation3':
        array = np.asarray(array, dtype=np.float64)

        u, _, vh = np.linalg.svd(array)
        array = np.dot(u, vh)

        if np.linalg.det(array) < 0:
            u[:, -1] *= -1
            array = np.dot(u, vh)

        return cls.new_unchecked(array)

    @classmethod
    def from_euler_angles(cls, xyz: Union[List[List[float]], np.ndarray]) -> 'Rotation3':
        if isinstance(xyz, list):
            if len(xyz) != 3:
                raise ValueError("List must contain exactly three numbers.")
        elif isinstance(xyz, np.ndarray):
            if xyz.shape != (3,):
                raise ValueError("Array must contain exactly three numbers.")
        else:
            raise TypeError("Input must be either a list of three numbers or a numpy array of three numbers.")

        roll, pitch, yaw = xyz

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        rotation_matrix = R_z @ R_y @ R_x

        return cls(rotation_matrix)

    def inverse(self) -> 'Rotation3':
        return self.new_unchecked(self.array.T)

    def map_point(self, v: V3) -> 'V3':
        return V3(self.array@v.array)

    def __repr__(self) -> str:
        return f"Rotation3(\n{np.array2string(self.array)}\n)"

    def __str__(self) -> str:
        return f"Rotation3(\n{np.array2string(self.array)}\n)"



