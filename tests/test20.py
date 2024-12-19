import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    TestFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend

f = TestFunction()
d = DerivativeMethodFD()

fe = FunctionEngine(f, d, Backend.PyTorch, jit_compile_f=True, jit_compile_d=False)
print(fe.call(np.array([1.0])))
