import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodReverseADJax, DerivativeMethodForwardADJax
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    TestFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend
import time

f = TestFunction()
d = DerivativeMethodForwardADJax(f)

fe = FunctionEngine(f, d, jit_compile_f=False, jit_compile_d=False)

fe.derivative(np.array([0.0]))

start = time.time()
fx, dfdx = fe.derivative(np.array([0.0]))
end = time.time() - start

print(fx)
print(dfdx)
print(end)
