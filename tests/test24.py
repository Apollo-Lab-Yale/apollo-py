import time

import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodForwardADJax
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend

f = BenchmarkFunction(80, 1, 10000)
d = DerivativeMethodForwardADJax()
fe = FunctionEngine(f, d)

x = np.random.uniform(-1, 1, (80, )).tolist()

start = time.time()
res = fe.derivative(x)
print(time.time() - start)

start = time.time()
res = fe.derivative(x)
print(time.time() - start)

