import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction

n = 3
f = BenchmarkFunction(n, 1, 10)

g = DerivativeMethodFD()

x = np.random.uniform(-1, 1, (n, ))
fx = f.call(x)
r = 3

Delta_x = np.random.uniform(-5, 5, (n, r))
Delta_f = np.zeros((1, r))

for i in range(r):
    delta_x_i = Delta_x[:, i]
    delta_f_i = (f.call(x + 0.00001*delta_x_i) - fx) / 0.00001
    Delta_f[:, i] = delta_f_i

res1 = (np.linalg.inv(Delta_x.T) @ Delta_f.T).T
res2 = g.derivative(f, x)

print(res1)
print(res2)

####

e1 = np.zeros((n, ))
e1[0] = 0.001

