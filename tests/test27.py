import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodWASP3, DerivativeMethodFD, get_random_walk
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend

b = Backend.Numpy
f = BenchmarkFunction(2, 2, 1000)
d = DerivativeMethodWASP3(2, 2, b)
fe = FunctionEngine(f, d, backend=b)
fe2 = FunctionEngine(f, DerivativeMethodFD(), backend=b)

w = get_random_walk(2, 100, 0.2)

for i in range(20):
    res = fe.derivative(w[i])
    print(res)
    res = fe2.derivative(w[i])
    print(res)
    print('---')


