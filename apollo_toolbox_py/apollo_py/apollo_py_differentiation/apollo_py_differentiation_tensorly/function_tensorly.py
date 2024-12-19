from abc import ABC, abstractmethod
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
        return [tl.sin(x[0])]

    def input_dim(self):
        return 1

    def output_dim(self):
        return 1
