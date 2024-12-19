from abc import ABC, abstractmethod
from typing import List

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    FunctionTensorly
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2


class DerivativeMethodTensorly(ABC):
    @abstractmethod
    def allowable_backends(self) -> List[Backend]:
        pass

    @abstractmethod
    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        pass

    def derivative(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        assert x.shape == (f.input_dim(),)
        fx, dfdx = self.derivative_raw(f, x)
        assert fx.shape == (f.output_dim(),)
        assert dfdx.shape == (f.output_dim(), f.input_dim())
        return fx, dfdx


class DerivativeMethodFD(DerivativeMethodTensorly):
    def __init__(self, epsilon=0.000001):
        self.epsilon = epsilon

    def allowable_backends(self) -> List[Backend]:
        return [Backend.Numpy, Backend.JAX, Backend.PyTorch]

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        pass
