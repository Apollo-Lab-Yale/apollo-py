import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodFD, DerivativeMethodReverseADPytorch
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
import tensorly as tl

n = 3
f = BenchmarkFunction(n, 1, 10)

g = DerivativeMethodFD()
fe = FunctionEngine(f, DerivativeMethodReverseADPytorch())

x = np.random.uniform(-1, 1, (n,))
fx = f.call(x)
# e1 = np.zeros((n, ))
# e1[0] = 1.0
ep = 0.1
# e1 *= ep
# fxe1 = f.call(x + e1)
r = 3

rxs = []
for i in range(r):
    # rxs.append(x + np.random.uniform(-0.00000001, 0.00000001, (n,)))
    tmp = np.zeros((n, ))
    tmp[i] = ep
    rxs.append(x + tmp)

rfs = []
for i in range(r):
    rfs.append(f.call(rxs[i]))

Delta_x_1 = np.zeros((n, r))
for i in range(n):
    Delta_x_1[:, i] = rxs[i] - x

Delta_f_1 = np.zeros((1, r))
for i in range(n):
    Delta_f_1[:, i] = rfs[i] - fx


DT = np.linalg.pinv(Delta_x_1.T) @ Delta_f_1.T
a1 = DT.T
print(a1)

b1 = fe.derivative(x)
print(b1)

tl.set_backend('numpy')
c1 = g.derivative(f, x)
print(c1)

#####

e1 = np.zeros((n, ))
e1[0] = 1.0
e1 *= ep
fxe1 = f.call(x + e1)

rxs = []
rxs.append(x)
for i in range(r):
    if i == 0:
        continue
    # rxs.append(x + np.random.uniform(-0.00000001, 0.00000001, (n,)))
    tmp = np.zeros((n, ))
    tmp[i] = ep
    rxs.append(x + tmp)

rfs = []
for i in range(r):
    rfs.append(f.call(rxs[i]))

Delta_x_1 = np.zeros((n, r))
for i in range(n):
    Delta_x_1[:, i] = rxs[i] - (x + e1)

Delta_f_1 = np.zeros((1, r))
for i in range(n):
    Delta_f_1[:, i] = rfs[i] - fxe1

print()

DT = np.linalg.pinv(Delta_x_1.T) @ Delta_f_1.T
a2 = DT.T
print(a2)

b2 = fe.derivative(x + e1)
print(b2)

tl.set_backend('numpy')
c2 = g.derivative(f, x + e1)
print(c2)

print()

print(a2 - a1)
print(b2 - b1)
print(c2 - c1)

print()

print((a2 - a1)/ep)
print((b2 - b1)/ep)
print((c2 - c1)/ep)
