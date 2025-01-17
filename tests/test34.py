from typing import List

import numpy as np
import tensorly as tl

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodReverseADPytorch
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction, FunctionTensorly


n = 3
f = BenchmarkFunction(n, 1, 10)

g = DerivativeMethodFD()

x = np.random.uniform(-1, 1, (n, ))
fx = f.call(x)
e1 = np.zeros((n, ))
e1[0] = 1.0
ep = 0.001
e1 *= ep
fxe1p = f.call(x + e1)
fxe1n = f.call(x - e1)
r = 3

rxs = []
for i in range(r):
    rxs.append(x + np.random.uniform(-0.1, 0.1, (n, )))

rfs = []
for i in range(r):
    rfs.append(f.call(rxs[i]))

Delta_x_1 = np.zeros((n, r))
for i in range(n):
    Delta_x_1[:, i] = rxs[i] - (x - e1)

Delta_f_1 = np.zeros((1, r))
for i in range(n):
    Delta_f_1[:, i] = rfs[i] - fxe1n

DT = np.linalg.pinv(Delta_x_1.T) @ Delta_f_1.T
a1 = DT.T
b1 = g.derivative(f, x - e1)

######

rxs = []
for i in range(r):
    rxs.append(x + np.random.uniform(-0.1, 0.1, (n, )))

rfs = []
for i in range(r):
    rfs.append(f.call(rxs[i]))

Delta_x_2 = np.zeros((n, r))
for i in range(n):
    Delta_x_2[:, i] = rxs[i] - (x + e1)

Delta_f_2 = np.zeros((1, r))
for i in range(n):
    Delta_f_2[:, i] = rfs[i] - fxe1p

DT = np.linalg.pinv(Delta_x_2.T) @ Delta_f_2.T
a2 = DT.T
b2 = g.derivative(f, x + e1)

print(a1)
print(b1)

print()

print(a2)
print(b2)

print()

print(a2 - a1)
print(b2 - b1)

print()

print((a2 - a1)/(2.0*ep))
print((b2 - b1)/(2.0*ep))



