from apollo_toolbox_py.apollo_py.apollo_py_derivatives.derivative_engine import JitCompileMode, ADMode
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.fd_derivatives import FDDerivativeEngine
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.jax_derivatives import JaxDerivativeEngine
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.test_functions import random_vector, \
    SimpleNeuralNetwork, MatrixMultiply
import jax
import time

from apollo_toolbox_py.apollo_py.apollo_py_derivatives.wasp_derivatives import WASPDerivativeEngine

jax.config.update("jax_enable_x64", True)

m = 1
n = 3

# f = SimpleNeuralNetwork.new_random(n, 10, m, 0)
f = MatrixMultiply.new_random(m, n, 0)

v = random_vector(n, -1.0, 1.0)
t = time.time()
for i in range(1000):
    res = f.call(v)
print(time.time() - t)

d = JaxDerivativeEngine(f.call, n, m, JitCompileMode.Jax, True, ADMode.Forward)
der = d.derivative(v)
t = time.time()
for i in range(1000):
    der = d.derivative(v)
print(time.time() - t)

d = JaxDerivativeEngine(f.call, n, m, JitCompileMode.Jax, True, ADMode.Reverse)
der = d.derivative(v)
t = time.time()
for i in range(1000):
    der = d.derivative(v)
print(time.time() - t)

d = WASPDerivativeEngine(f.call, n, m, JitCompileMode.Jax)
der = d.derivative(v)
t = time.time()
for i in range(1000):
    der = d.derivative(v)
print(time.time() - t)

d = FDDerivativeEngine(f.call, n, m, JitCompileMode.Jax)
t = time.time()
for i in range(1000):
    der = d.derivative(v)
print(time.time() - t)


