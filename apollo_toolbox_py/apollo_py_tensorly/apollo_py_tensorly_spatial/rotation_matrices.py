from typing import Union, List

import numpy as np
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2, Device, DType
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M3


class Rotation3(M3):
    def __init__(self, array: Union[List[List[float]], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        super().__init__(array, device, dtype)
        if not T2.allclose(self.array @ self.array.T, np.eye(3), rtol=1e-7, atol=1e-7):
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

    def __repr__(self) -> str:
        return f"Rotation3(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"Rotation3(\n{self.array}\n)"