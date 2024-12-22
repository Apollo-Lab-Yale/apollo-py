import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodReverseADJax
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
import tensorly as tl
import time

from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend, Device

f = BenchmarkFunction(10, 10, 1000)
d = DerivativeMethodReverseADJax()
fe = FunctionEngine(f, d, backend=Backend.JAX, device=Device.CPU, jit_compile_f=True, jit_compile_d=True)

start = time.time()
res = fe.derivative(tl.tensor(np.random.uniform(-1, 1, (10, ))))
duration = time.time() - start
print(duration)
print('got here')

start = time.time()
for i in range(1):
    res = fe.derivative(np.random.uniform(-1, 1, (10, )))
duration = time.time() - start
print(res)
print(duration)
