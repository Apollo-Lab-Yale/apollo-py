from abc import ABC, abstractmethod
from typing import List

import jax

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
    def default_backend(self) -> Backend:
        pass

    @abstractmethod
    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        pass

    def derivative(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        assert x.shape == (f.input_dim(),)
        fx, dfdx = self.derivative_raw(f, x)
        assert fx.shape == (f.output_dim(),)
        assert dfdx.shape == (f.output_dim(), f.input_dim()), 'shape is {}'.format(dfdx.shape)
        return fx, dfdx


class DerivativeMethodFD(DerivativeMethodTensorly):
    def __init__(self, epsilon=0.000001):
        self.epsilon = epsilon

    def allowable_backends(self) -> List[Backend]:
        return [Backend.Numpy, Backend.JAX, Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.Numpy

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        fx = f.call(x)
        dfdx = tl.zeros((f.output_dim(), f.input_dim()), device=getattr(x, 'device', None), dtype=x.dtype)

        for i in range(f.input_dim()):
            delta_x = tl.zeros(f.input_dim(), device=getattr(x, 'device', None), dtype=x.dtype)
            delta_x = T2.set_and_return(delta_x, i, self.epsilon)
            x_delta_x = x + delta_x
            fh = f.call(x_delta_x)
            col = (fh - fx) / self.epsilon
            dfdx = T2.set_and_return(dfdx, (slice(None), i), col)

        return fx, dfdx


class DerivativeMethodReverseADJax(DerivativeMethodTensorly):
    def __init__(self, f: FunctionTensorly):
        self.jac_fn = jax.jacrev(f.call)

    def allowable_backends(self) -> List[Backend]:
        return [Backend.JAX]

    def default_backend(self) -> Backend:
        return Backend.JAX

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        fx = f.call(x)
        dfdx = self.jac_fn(x)

        return fx, dfdx


class DerivativeMethodForwardADJax(DerivativeMethodTensorly):
    def __init__(self, f: FunctionTensorly):
        self.jac_fn = jax.jacfwd(f.call)

    def allowable_backends(self) -> List[Backend]:
        return [Backend.JAX]

    def default_backend(self) -> Backend:
        return Backend.JAX

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        fx = f.call(x)
        dfdx = self.jac_fn(x)

        return fx, dfdx


class DerivativeMethodReverseADPytorch(DerivativeMethodTensorly):
    def allowable_backends(self) -> List[Backend]:
        return [Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.PyTorch

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> (tl.tensor, tl.tensor):
        dfdx = tl.zeros(f.output_dim(), f.input_dim(), device=getattr(x, 'device', None), dtype=x.dtype)

        x.requires_grad = True
        fx = f.call(x)
        for i in range(f.input_dim()):
            if x.grad is not None:
                x.grad.zero_()

            fx[i].backward(retain_graph=True)
            col = x.grad.clone()
            dfdx[:, i] = col

        return fx, dfdx
