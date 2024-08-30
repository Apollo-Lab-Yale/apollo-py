import time
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.derivative_engine import FunctionMode, ADMode
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.fd_derivatives import FDDerivativeEngine
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.jax_derivatives import JaxDerivativeEngine
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.wasp_derivatives import WASPDerivativeEngine, JitCompileMode
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def f(x):
    return [x[0] ** 2 + x[1] ** 2, x[1]**2 + x[2]**2]


w = WASPDerivativeEngine(f, 3, 2, JitCompileMode.Jax, 1.0)
res = w.derivative([5., 6., 7.])
print(res)

res = w.derivative([5.2, 6.1, 7.1])
print(res)

res = w.derivative([5.7, 6.2, 7.4])
print(res)

res = w.derivative([5.7, 6.2, 7.4])
print(res)

w = JaxDerivativeEngine(f, 3, 2, JitCompileMode.Jax, True, ADMode.Reverse)
start = time.time()
w.derivative([5.7, 6.2, 7.4])
end = time.time()
print(end - start)

start = time.time()
for i in range(1000):
    w.call_numpy([5.7, 6.2, 7.4])
end = time.time()
print(end - start)

start = time.time()
for i in range(1000):
    w.derivative([5.7, 6.2, 7.4])
end = time.time()
print(end - start)

