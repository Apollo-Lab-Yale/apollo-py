import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodReverseADJax, DerivativeMethodForwardADJax, DerivativeMethodReverseADPytorch
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    TestFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend
import time
import tensorly as tl

f = TestFunction()
d = DerivativeMethodReverseADPytorch()

fe = FunctionEngine(f, d, jit_compile_f=False, jit_compile_d=False)

fe.derivative(np.array([0.0, 1.0]))

start = time.time()
dfdx = fe.derivative(np.array([0.0, 1.0]))
end = time.time() - start

print(dfdx)
print(end)
