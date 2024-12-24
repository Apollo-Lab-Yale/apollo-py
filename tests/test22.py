import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodReverseADJax, DerivativeMethodWASP, DerivativeMethodForwardADJax, \
    DerivativeMethodReverseADPytorch
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
import tensorly as tl
import time

from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend, Device, DType

f = BenchmarkFunction(10, 10, 10000)
d = DerivativeMethodWASP(10, 10, Backend.Numpy)
# d = DerivativeMethodReverseADPytorch()
# d = DerivativeMethodFD()
fe = FunctionEngine(f, d, device=Device.CPU, dtype=DType.Float64, jit_compile_f=False, jit_compile_d=False)

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
