import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodWASP4, DerivativeMethodWASP3, DerivativeMethodWASP2, DerivativeMethodFD, get_random_walk, \
    DerivativeMethodWASP, DerivativeMethodWASP5
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend

n = 20
m = 10

ff2 = 0.0
ff3 = 0.0
ff4 = 0.0
ff5 = 0.0
ff6 = 0.0
ff7 = 0.0
ff8 = 0.0

for o in range(30):
    f = BenchmarkFunction(n, m, 1000)
    # fe = FunctionEngine(f, DerivativeMethodFD())
    fe2 = FunctionEngine(f, DerivativeMethodWASP5(n, m, Backend.Numpy, d_ell=0.2, d_theta=0.2))
    fe3 = FunctionEngine(f, DerivativeMethodWASP4(n, m, Backend.Numpy, d_ell=0.2, d_theta=0.2, delta_x_scaling=0.2))
    fe4 = FunctionEngine(f, DerivativeMethodWASP2(n, m, Backend.Numpy, d_ell=0.2, d_theta=0.2))
    fe5 = FunctionEngine(f, DerivativeMethodWASP(n,  m, Backend.Numpy, d_ell=0.2, d_theta=0.2))
    fe6 = FunctionEngine(f, DerivativeMethodWASP4(n, m, Backend.Numpy, d_ell=0.2, d_theta=0.2, delta_x_scaling=0.5))
    fe7 = FunctionEngine(f, DerivativeMethodWASP4(n, m, Backend.Numpy, d_ell=0.2, d_theta=0.2, delta_x_scaling=1.0))
    fe8 = FunctionEngine(f, DerivativeMethodWASP4(n, m, Backend.Numpy, d_ell=0.2, d_theta=0.2, delta_x_scaling=0.1))

    w = get_random_walk(n, 100, 0.2)

    for i in range(100):
        # print(fe.call(w[i]))
        # print(fe.derivative(w[i]))
        print(fe2.derivative(w[i]))
        print(fe3.derivative(w[i]))
        print(fe4.derivative(w[i]))
        print(fe5.derivative(w[i]))
        print(fe6.derivative(w[i]))
        print(fe7.derivative(w[i]))
        print(fe8.derivative(w[i]))
        ff2 += float(fe2.d.num_f_calls)
        ff3 += float(fe3.d.num_f_calls)
        ff4 += float(fe4.d.num_f_calls)
        ff5 += float(fe5.d.num_f_calls)
        ff6 += float(fe6.d.num_f_calls)
        ff7 += float(fe7.d.num_f_calls)
        ff8 += float(fe8.d.num_f_calls)
        print('---')

print('wasp 5: ', ff2 / (30.0 * 100.0))
print('wasp 4 (0.2): ', ff3 / (30.0 * 100.0))
print('wasp 2: ', ff4 / (30.0 * 100.0))
print('wasp 1: ', ff5 / (30.0 * 100.0))
print('wasp 4 (0.5): ', ff6 / (30.0 * 100.0))
print('wasp 4 (1.0): ', ff7 / (30.0 * 100.0))
print('wasp 4 (0.1): ', ff8 / (30.0 * 100.0))

