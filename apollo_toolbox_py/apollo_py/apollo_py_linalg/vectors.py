from typing import Union, List, TypeVar
import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_array.apollo_py_array import ApolloPyArray, ApolloPyArrayBackend

# __all__ = ['V', 'V3', 'V6']

B = TypeVar('B', bound='ApolloPyArrayBackend')


class V:
    def __init__(self, array: Union[List[float], np.ndarray, ApolloPyArray], backend: B = None):
        if isinstance(array, ApolloPyArray):
            self.array = array
            return

        self.array = ApolloPyArray.new_with_backend(array, backend)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def dot(self, other):
        return self.array.dot(other.array)

    def norm(self):
        return self.array.p_norm(2)

    def normalize(self):
        return self.__class__(self.array / self.norm())

    def __add__(self, other):
        return self.__class__(self.array + other)

    def __sub__(self, other):
        return self.__class__(self.array - other)

    def __mul__(self, scalar):
        return self.__class__(self.array * scalar)

    def __rmul__(self, scalar):
        return self.__class__(self.array * scalar)

    def __neg__(self):
        return self.__class__(-self.array)

    def __truediv__(self, scalar):
        return self.__class__(self.array / scalar)

    def __rtruediv__(self, scalar):
        return self.__class__(scalar / self.array)

    def magnitude(self):
        return self.norm()

    def unit(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Zero vector has no direction.")
        return self.__class__(self.array / norm)

    def __repr__(self) -> str:
        return f"V(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"V(\n{self.array}\n)"