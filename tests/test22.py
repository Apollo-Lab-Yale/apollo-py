import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    close_enough, get_tangent_matrix, WASPCache, DerivativeMethodWASP
import tensorly as tl

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    TestFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2, Backend

b = Backend.Numpy
f = TestFunction()
d = DerivativeMethodWASP(f.input_dim(), f.output_dim(), b)
fe = FunctionEngine(f, d, b, jit_compile_f=True)

print(fe.derivative([1., 2.]))
print(fe.d.num_f_calls)

print(fe.derivative([1., 2.]))
print(fe.d.num_f_calls)

print(fe.derivative([1., 2.]))
print(fe.d.num_f_calls)
