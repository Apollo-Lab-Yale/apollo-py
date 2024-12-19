import numba

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
except ImportError:
    jax = None

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodTensorly
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    FunctionTensorly
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend, Device, DType
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2


class FunctionEngine:
    def __init__(self,
                 f: FunctionTensorly,
                 d: DerivativeMethodTensorly,
                 backend: Backend = Backend.Numpy,
                 device: Device = Device.CPU,
                 dtype: DType = DType.Float64,
                 jit_compile_f: bool = False,
                 jit_compile_d: bool = False):
        assert d.allowable_backends().__contains__(backend), 'Backend {} not allowable for derivative method {}.  Legal methods are {}'.format(backend, d, d.allowable_backends())

        self.backend = backend
        self.device = device
        self.dtype = dtype

        tl.set_backend(backend.to_string())

        def f_call(x):
            return f.call(x)

        def d_call(x):
            return d.derivative(f, x)

        self.f_call = f_call
        self.d_call = d_call

        if jit_compile_f:
            if backend == Backend.Numpy:
                print('WARNING: jit compilation not currently supported with Numpy backend')
                # self.f_call = numba.jit(self.f_call)
            elif backend == Backend.JAX:
                self.f_call = jax.jit(self.f_call)
            elif backend == Backend.PyTorch:
                self.f_call = torch.jit.trace(self.f_call, torch.randn((f.input_dim())))
            else:
                raise NotImplementedError(f"Backend {backend} is not supported.")

        if jit_compile_d:
            if backend == Backend.Numpy:
                pass
                # self.d_call = numba.jit(self.d_call)
            elif backend == Backend.JAX:
                self.d_call = jax.jit(self.d_call)
            elif backend == Backend.PyTorch:
                self.d_call = torch.jit.trace(self.d_call, torch.randn((f.input_dim())))
            else:
                raise NotImplementedError(f"Backend {backend} is not supported.")

    def call(self, x: tl.tensor) -> tl.tensor:
        tl.set_backend(self.backend.to_string())
        x = tl.to_numpy(x)
        x = T2.new(x, self.device, self.dtype)
        return self.f_call(x)

    def derivative(self, x: tl.tensor) -> (tl.tensor, tl.tensor):
        tl.set_backend(self.backend.to_string())
        x = tl.to_numpy(x)
        x = T2.new(x, self.device, self.dtype)
        return self.d_call(x)