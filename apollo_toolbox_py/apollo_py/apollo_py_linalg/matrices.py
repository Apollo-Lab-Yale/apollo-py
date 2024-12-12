from typing import List, Union, TypeVar
import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_array import ApolloPyArray

__all__ = ['M', 'M3']

B = TypeVar('B', bound='ApolloPyArrayBackend')


class M:
    def __init__(self, array: Union[List[List[float]], np.ndarray, ApolloPyArray], backend: B = None):
        if isinstance(array, ApolloPyArray):
            self.array = array
            return

        self.array = ApolloPyArray.new_with_backend(array, backend)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def transpose(self):
        return self.__class__(self.array.T)

    @property
    def T(self):
        return self.transpose()

    def det(self):
        return self.array.det()

    def rank(self) -> int:
        return self.array.matrix_rank()

    def inv(self):
        return self.__class__(self.array.inv())

    def pinv(self):
        return self.__class__(self.array.pinv())

    def dot(self, other: 'M') -> 'M':
        return self.__class__(self.array.dot(other.array))

    def trace(self) -> ApolloPyArray:
        return self.array.trace()

    def __add__(self, other: 'M') -> 'M':
        return self.__class__(self.array + other.array)

    def __sub__(self, other: 'M') -> 'M':
        return self.__class__(self.array - other.array)

    def __mul__(self, scalar) -> 'M':
        return self.__class__(self.array * scalar)

    def __truediv__(self, scalar) -> 'M':
        return self.__class__(self.array / scalar)

    def __rtruediv__(self, scalar) -> 'M':
        return self.__class__(scalar / self.array)

    def __matmul__(self, other: 'M') -> 'M':
        return self.__class__(self.array @ other.array)

    def __repr__(self) -> str:
        return f"Matrix(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"Matrix(\n{self.array}\n)"


class M3(M):
    def __init__(self, array: Union[List[List[float]], np.ndarray, ApolloPyArray], backend: B = None):
        super().__init__(array, backend)
        if self.array.shape != (3, 3):
            raise ValueError("Matrix3 must be a 3x3 matrix.")

    def __repr__(self) -> str:
        return f"Matrix3(\n{self.array}\n)"

    def __str__(self) -> str:
        return f"Matrix3(\n{self.array}\n)"
