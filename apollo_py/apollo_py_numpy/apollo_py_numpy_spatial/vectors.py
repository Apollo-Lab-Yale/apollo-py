from typing import Union, List
import numpy as np


class V3:
    def __init__(self, array: Union[List[float], np.ndarray]):
        self.array = np.asarray(array, dtype=np.float64)
        if self.array.shape != (3,):
            raise ValueError("V3 must be a 3-vector.")

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def to_lie_alg_so3(self):
        from apollo_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.so3 import LieAlgSO3
        LieAlgSO3.from_euclidean_space_element(self.array)

    def __repr__(self) -> str:
        return f"V3(\n{np.array2string(self.array)}\n)"

    def __str__(self) -> str:
        return f"V3(\n{np.array2string(self.array)}\n)"


class V6:
    def __init__(self, array: Union[List[float], np.ndarray]):
        self.array = np.asarray(array, dtype=np.float64)
        if self.array.shape != (6,):
            raise ValueError("V6 must be a 6-vector.")

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self) -> str:
        return f"V6(\n{np.array2string(self.array)}\n)"

    def __str__(self) -> str:
        return f"V6(\n{np.array2string(self.array)}\n)"
