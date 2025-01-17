import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.hessian_method_tensorly import \
    HessianMethodElementwiseFD, HessianMethodGradientwiseFD

f = BenchmarkFunction(3, 1, 1000)

g = DerivativeMethodFD()
h = HessianMethodElementwiseFD()
h2 = HessianMethodGradientwiseFD(g)

x = np.random.uniform(-1, 1, (3,))
dx = 0.001 * np.array([1., 0., 0.])
x2 = x + dx

# res = h.hessian(f, x)
# print(res)
#
# print()
#
# res = h2.hessian(f, x)
# print(res)

d = g.derivative(f, x)
print(d)

ff = f.call(x)
ff2 = f.call(x2)

# print(ff)
# print(f.call(x2))
# print(ff + d@dx)

print(ff2 - ff)
print(d@dx)

