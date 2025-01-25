import time

import jax
import jax.numpy as jnp

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    get_random_walk
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction2JAX

n = 10
w = get_random_walk(n, 1000, 0.05)

f = BenchmarkFunction2JAX(n, 1, 100)

start = time.time()
g = jax.jit(jax.jacfwd(f.call_raw))
for i in range(1000):
    g(jnp.array(w[i]))

print(time.time() - start)
