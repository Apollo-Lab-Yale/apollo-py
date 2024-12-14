from tensorly import backend as T
import numpy as np
import jax.numpy as jnp


class ExtraTensorlyFunctions:

    @staticmethod
    def det(tl_tensor):
        b = T.get_backend()
        if b == 'numpy':
            return np.linalg.det(tl_tensor)
        elif b == 'jax':
            return jnp.linalg.det(tl_tensor)
        elif b == 'pytorch':
            return tl_tensor.det()
        else:
            raise ValueError(f'Backend {b} is not supported.')
