import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodWASP3, DerivativeMethodFD
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend

b = Backend.Numpy
f = BenchmarkFunction(2, 3, 1000)
d = DerivativeMethodWASP3(2, 3, b)
fe = FunctionEngine(f, d, backend=b)
fe2 = FunctionEngine(f, DerivativeMethodFD(), backend=b)



