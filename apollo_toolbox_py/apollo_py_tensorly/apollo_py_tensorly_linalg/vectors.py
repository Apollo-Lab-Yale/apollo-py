from typing import Union, List
import tensorly as tl
from tensorly import backend as T
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2, Device, DType

import numpy as np


class V:
    def __init__(self, array: Union[List[float], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        if tl.is_tensor(array):
            self.array = array
            return

        self.array = T2.new(array, device, dtype)
        assert len(self.array.shape) == 1

    def __getitem__(self, item):
        return self.array[item]

    def set_and_return(self, key, value):
        return self.__class__(T2.set_and_return(self.array, key, value))

    def dot(self, other):
        if isinstance(other, V):
            other = other.array
        return np.dot(self.array, other)

    def norm(self, ord=2):
        return tl.norm(self.array, order=ord)

    def normalize(self):
        return self.__class__(self.array / self.norm())

    def __add__(self, other):
        if isinstance(other, V):
            other = other.array
        return self.__class__(self.array + other)

    def __sub__(self, other):
        if isinstance(other, V):
            other = other.array
        return self.__class__(self.array - other)

    def __mul__(self, scalar):
        return self.__class__(self.array * scalar)

    def __rmul__(self, scalar):
        return self.__class__(self.array * scalar)

    def __neg__(self):
        return self.__class__(-self.array)

    def __truediv__(self, scalar):
        if scalar == 0:
            raise ValueError("Division by zero is not allowed.")
        return self.__class__(self.array / scalar)

    def __rtruediv__(self, scalar):
        return self.__class__(scalar / self.array)

    def magnitude(self):
        return np.linalg.norm(self.array)

    def unit(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Zero vector has no direction.")
        return self.__class__(self.array / norm)

    def __repr__(self) -> str:
        return f"V(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"V(\n{self.array}\n)"


class V3(V):
    def __init__(self, array: Union[List[float], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        super().__init__(array, device, dtype)
        if self.array.shape != (3,):
            raise ValueError("V3 must be a 3-vector.")

    def cross(self, other: 'V3') -> 'V3':
        if not isinstance(other, V3):
            raise TypeError("Cross product requires another V3 vector.")
        return V3(T2.cross(self.array, other.array))

    def __repr__(self) -> str:
        return f"V3(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"V3(\n{self.array}\n)"


class V6(V):
    def __init__(self, array: Union[List[float], np.ndarray], device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        super().__init__(array, device, dtype)
        if self.array.shape != (6,):
            raise ValueError("V6 must be a 6-vector.")

    def __repr__(self) -> str:
        return f"V6(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"V6(\n{self.array}\n)"
