from typing import Union, List

import numpy as np

from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.vectors import V3


class Quaternion:
    def __init__(self, wxyz_array: Union[List[float], np.ndarray]):
        self.array = np.asarray(wxyz_array, dtype=np.float64)
        if self.array.shape != (4,):
            raise ValueError("Quaternion must be a 4-vector.")

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    @property
    def w(self):
        return self.array[0]

    @property
    def x(self):
        return self.array[1]

    @property
    def y(self):
        return self.array[2]

    @property
    def z(self):
        return self.array[3]

    def conjugate(self) -> 'Quaternion':
        w, x, y, z = self.array
        return Quaternion([w, -x, -y, -z])

    def inverse(self) -> 'Quaternion':
        conjugate = self.conjugate()
        norm_sq = np.linalg.norm(self.array) ** 2
        return Quaternion(conjugate.array / norm_sq)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        w1, x1, y1, z1 = self.array
        w2, x2, y2, z2 = other.array

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion([w, x, y, z])

    def __matmul__(self, other: 'Quaternion') -> 'Quaternion':
        return self * other

    def __repr__(self) -> str:
        return f"Quaternion(\n{np.array2string(self.array)}\n)"

    def __str__(self) -> str:
        return f"Quaternion(\n{np.array2string(self.array)}\n)"


class UnitQuaternion(Quaternion):
    def __init__(self, wxyz_array: Union[List[float], np.ndarray]):
        super().__init__(wxyz_array)
        if not np.isclose(np.linalg.norm(self.array), 1.0, rtol=1e-7, atol=1e-7):
            raise ValueError("Unit quaternion must be unit length.")

    @classmethod
    def new_unchecked(cls, wxyz_array: Union[List[float], np.ndarray]) -> 'UnitQuaternion':
        out = cls.__new__(cls)
        out.array = np.asarray(wxyz_array)
        return out

    @classmethod
    def new_normalize(cls, wxyz_array: Union[List[float], np.ndarray]) -> 'UnitQuaternion':
        out = cls.new_unchecked(wxyz_array)
        out.array /= np.linalg.norm(out.array)
        return out

    @classmethod
    def from_euler_angles(cls, xyz: Union[List[float], np.ndarray]) -> 'UnitQuaternion':
        if isinstance(xyz, list):
            if len(xyz) != 3:
                raise ValueError("List must contain exactly three numbers.")
        elif isinstance(xyz, np.ndarray):
            if xyz.shape != (3,):
                raise ValueError("Array must contain exactly three numbers.")
        else:
            raise TypeError("Input must be either a list of three numbers or a numpy array of three numbers.")

        cy = np.cos(xyz[2] * 0.5)
        sy = np.sin(xyz[2] * 0.5)
        cp = np.cos(xyz[1] * 0.5)
        sp = np.sin(xyz[1] * 0.5)
        cr = np.cos(xyz[0] * 0.5)
        sr = np.sin(xyz[0] * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls([w, x, y, z])

    def inverse(self) -> 'UnitQuaternion':
        return self.conjugate()

    def to_rotation_matrix(self) -> 'Rotation3':
        from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.rotation_matrices import Rotation3
        w, x, y, z = self.array
        matrix = [
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ]
        return Rotation3.new_unchecked(matrix)

    def map_point(self, v: V3) -> 'V3':
        qv = Quaternion([0.0, v.array[0], v.array[1], v.array[2]])
        res = self@qv@self.conjugate()
        return V3([res[1], res[2], res[3]])

    def to_lie_group_h1(self) -> 'LieGroupH1':
        from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.h1 import LieGroupH1
        return LieGroupH1(self.array)

    def __mul__(self, other: Union['UnitQuaternion', 'Quaternion']) -> Union['UnitQuaternion', 'Quaternion']:
        tmp = super().__mul__(other)
        if isinstance(other, UnitQuaternion):
            return UnitQuaternion.new_unchecked(tmp.array)
        else:
            return tmp

    def __matmul__(self, other: Union['UnitQuaternion', 'Quaternion']) -> Union['UnitQuaternion', 'Quaternion']:
        return self * other

    def __repr__(self) -> str:
        return f"UnitQuaternion(\n{np.array2string(self.array)}\n)"

    def __str__(self) -> str:
        return f"UnitQuaternion(\n{np.array2string(self.array)}\n)"
