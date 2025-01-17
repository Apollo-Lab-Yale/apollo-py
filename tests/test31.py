import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction

n = 3
r = 5
ep = 0.001
f = BenchmarkFunction(n, 1, 10)
d = DerivativeMethodFD()

x = np.random.uniform(-1, 1, (n,))
e_1 = np.zeros((n, ))
e_1[0] = 1.0
e_1_reshape = e_1.reshape(n, 1)

Delta_x = np.random.uniform(-0.001, 0.001, (n, r))

l = 2
Delta_x_l = Delta_x[:, l]
delta_f_1 = d.derivative(f, x + ep * e_1) @ Delta_x_l

# a = f.call(x) + d.derivative(f, x) @ (ep * e_1 + Delta_x_l)
# b = f.call(x) + d.derivative(f, x) @ (ep * e_1)

E_1 = np.zeros((n, r))
E_1[0, :] = ep

a = d.derivative(f, x) @ (E_1 + Delta_x)
b = d.derivative(f, x) @ E_1

print(a + b)
print(d.derivative(f, x - ep * e_1) @ Delta_x)
print(d.derivative(f, x) @ Delta_x)

# res = np.linalg.pinv(Delta_x.T) @ (a - b).T
# print(res.T)
# print(d.derivative(f, x + ep * e_1))
# print(d.derivative(f, x))


