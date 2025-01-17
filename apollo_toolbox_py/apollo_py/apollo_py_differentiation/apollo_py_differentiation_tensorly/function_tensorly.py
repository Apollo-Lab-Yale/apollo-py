from abc import ABC, abstractmethod
import random
from typing import List

from numba import jit
from numba.experimental import jitclass

from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Device, DType, Backend, ExtraBackend as T2
import tensorly as tl


class FunctionTensorly(ABC):

    def call(self, x: tl.tensor) -> tl.tensor:
        assert x.shape == (self.input_dim(),)
        out = T2.new_from_heterogeneous_array(self.call_raw(x))
        assert out.shape == (self.output_dim(),)
        return out

    @abstractmethod
    def call_raw(self, x: tl.tensor) -> List[tl.tensor]:
        pass

    @abstractmethod
    def input_dim(self):
        pass

    @abstractmethod
    def output_dim(self):
        pass


class TestFunction(FunctionTensorly):

    def call_raw(self, x: tl.tensor) -> List[tl.tensor]:
        return [tl.sin(x[0]), tl.cos(x[1])]

    def input_dim(self):
        return 2

    def output_dim(self):
        return 2


class BenchmarkFunction(FunctionTensorly):
    def __init__(self, n: int, m: int, num_operations: int):
        self.n = n
        self.m = m
        self.num_operations = num_operations
        self.r = []
        self.s = []
        for i in range(m):
            self.r.append([random.randint(0, n-1) for _ in range(num_operations + 1)])
            self.s.append([random.randint(0, 1) for _ in range(num_operations)])

    def call_raw(self, x: tl.tensor) -> List[tl.tensor]:
        out = []

        for i in range(self.m):
            rr = self.r[i]
            ss = self.s[i]
            tmp = x[rr[0]]
            for j in range(self.num_operations):
                if ss[j] == 0:
                    tmp = tl.sin(tmp * tl.cos(x[rr[j+1]]))
                elif ss[j] == 1:
                    tmp = tl.cos(tmp * tl.sin(x[rr[j + 1]]))
                else:
                    raise ValueError("Operation not supported")
            out.append(tmp)

        return out

    def input_dim(self):
        return self.n

    def output_dim(self):
        return self.m
