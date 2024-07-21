from typing import Union, List
import numpy as np


class M3:
    def __init__(self, array: Union[List[List[float]], np.ndarray]):
        self.array = np.asarray(array, dtype=np.float64)
        if self.array.shape != (3, 3):
            raise ValueError("Matrix3 must be a 3x3 matrix.")

    def __repr__(self) -> str:
        return f"Matrix3(\n{np.array2string(self.array)}\n)"

    def __str__(self) -> str:
        return f"Matrix3(\n{np.array2string(self.array)}\n)"
