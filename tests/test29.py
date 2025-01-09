from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodWASP, get_random_walk, DerivativeMethodFD, DerivativeMethodReverseADPytorch, \
    DerivativeMethodSPSA
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend
import numpy as np
import tensorly as tl

n = 4
m = 1

f = BenchmarkFunction(n, m, 1000)
fe = FunctionEngine(f, DerivativeMethodWASP(n, m, Backend.Numpy, d_ell=0.1, d_theta=0.1))
fe3 = FunctionEngine(f, DerivativeMethodFD())
fe2 = FunctionEngine(f, DerivativeMethodReverseADPytorch())
fe4 = FunctionEngine(f, DerivativeMethodSPSA())

w = get_random_walk(n, 1000, 0.0001)

print(fe4.derivative(w[0]))
print(fe4.derivative(w[1]))

print(fe3.derivative(w[0]))
print(fe3.derivative(w[1]))
