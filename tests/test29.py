from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodWASP2, DerivativeMethodWASP, get_random_walk, DerivativeMethodFD, DerivativeMethodReverseADPytorch
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

w = get_random_walk(n, 1000, 0.0001)

x = np.random.uniform(-1, 1, (n, ))
delta_x = np.zeros((n, ))
# delta_x[0] = 0.001
delta_x = np.random.uniform(-0.3, 0.3, (n, ))
# epsilon = 0.000001
x_plus_delta_x = x + 0.1*delta_x

# fe.d.fixed_i = 0
print(fe.derivative(w[0]))
print(tl.to_numpy(fe2.derivative(w[0])))
print(fe3.derivative(w[0]))

print()

print(fe.derivative(w[1]))
print(tl.to_numpy(fe2.derivative(w[1])))
print(fe3.derivative(w[1]))

print(fe.d.num_f_calls)
