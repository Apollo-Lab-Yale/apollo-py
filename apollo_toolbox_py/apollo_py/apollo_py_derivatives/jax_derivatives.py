__all__ = ['JaxDerivativeEngine']

import jax
import numpy as np
import jax.numpy as jnp

from apollo_toolbox_py.apollo_py.apollo_py_derivatives.derivative_engine import DerivativeEngine, JitCompileMode, \
    ADMode, FunctionMode


class JaxDerivativeEngine(DerivativeEngine):
    def __init__(self, f, n: int, m: int, jit_compile_f: JitCompileMode = JitCompileMode.DoNotJitCompile, ad_mode: ADMode = ADMode.Reverse):
        super().__init__(f, n, m)

        self.ad_mode = ad_mode

        if jit_compile_f == JitCompileMode.Numba:
            raise ValueError("JitCompileMode.Numba is not supported in JaxDerivativeEngine")

        if jit_compile_f == JitCompileMode.Jax:
            self.f = jax.jit(f)

    def derivative(self, x) -> np.ndarray:
        x = jnp.array(x)

        if self.ad_mode == ADMode.Forward:
            jac_fn = jax.jacfwd(self.call_jax)
        elif self.ad_mode == ADMode.Reverse:
            jac_fn = jax.jacrev(self.call_jax)
        else:
            raise ValueError("Unsupported ADMode")

        jacobian = jac_fn(x)
        return np.array(jacobian)
