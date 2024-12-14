import jax
import scipy
import tensorly as tl
import torch
from tensorly import backend as T
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from enum import Enum

__all__ = ['ExtraBackend', 'Device', 'DType']


class Device(Enum):
    CPU = 0
    MPS = 1
    CUDA = 2


class DType(Enum):
    Float32 = 0
    Float64 = 1


class ExtraBackend:
    @staticmethod
    def new(array, device: Device = Device.CPU, dtype: DType = DType.Float64):
        b = T.get_backend()
        if b == 'numpy':
            if dtype == DType.Float64:
                d = np.float64
            elif dtype == DType.Float32:
                d = np.float32
            else:
                raise ValueError('Unsupported dtype')
            return tl.tensor(array, dtype=d)
        elif b == 'jax':
            if dtype == DType.Float64:
                d = np.float64
            elif dtype == DType.Float32:
                d = np.float32
            else:
                raise ValueError('Unsupported dtype')

            if device == Device.CPU:
                de = jax.devices("cpu")[0]
            elif device == Device.MPS or device == Device.CUDA:
                try:
                    de = jax.devices("gpu")[0]
                except:
                    de = jax.devices("cpu")[0]
            else:
                raise ValueError('Unsupported device')

            return tl.tensor(array, dtype=d, device=de)
        elif b == 'pytorch':
            if dtype == DType.Float64:
                d = torch.float64
            elif dtype == DType.Float32:
                d = torch.float32
            else:
                raise ValueError('Unsupported dtype')

            if device == Device.CPU:
                de = 'cpu'
            elif device == Device.MPS:
                de = 'mps'
            elif device == Device.CUDA:
                de = 'cuda'
            else:
                raise ValueError('Unsupported device')

            return tl.tensor(array, dtype=d, device=de)

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

    @staticmethod
    def expm(tl_tensor):
        b = T.get_backend()
        if b == 'numpy':
            return scipy.linalg.expm(tl_tensor)
        elif b == 'jax':
            return jsp.linalg.expm(tl_tensor)
        elif b == 'pytorch':
            return tl_tensor.matrix_exp()
        else:
            raise ValueError(f'Backend {b} is not supported.')

    @staticmethod
    def cross(tl_tensor1, tl_tensor2):
        b = T.get_backend()
        if b == 'numpy':
            return np.cross(tl_tensor1, tl_tensor2)
        elif b == 'jax':
            return jnp.cross(tl_tensor1, tl_tensor2)
        elif b == 'pytorch':
            return torch.linalg.cross(tl_tensor1, tl_tensor2)
        else:
            raise ValueError(f'Backend {b} is not supported.')

    @staticmethod
    def inv(tl_tensor):
        b = T.get_backend()
        if b == 'numpy':
            return np.linalg.inv(tl_tensor)
        elif b == 'jax':
            return jnp.linalg.inv(tl_tensor)
        elif b == 'pytorch':
            return torch.linalg.inv(tl_tensor)
        else:
            raise ValueError(f'Backend {b} is not supported.')

    @staticmethod
    def pinv(tl_tensor):
        b = T.get_backend()
        if b == 'numpy':
            return np.linalg.pinv(tl_tensor)
        elif b == 'jax':
            return jnp.linalg.pinv(tl_tensor)
        elif b == 'pytorch':
            return torch.linalg.pinv(tl_tensor)
        else:
            raise ValueError(f'Backend {b} is not supported.')

    @staticmethod
    def matrix_rank(tl_tensor):
        b = T.get_backend()
        if b == 'numpy':
            return np.linalg.matrix_rank(tl_tensor)
        elif b == 'jax':
            return jnp.linalg.matrix_rank(tl_tensor)
        elif b == 'pytorch':
            return torch.linalg.matrix_rank(tl_tensor)
        else:
            raise ValueError(f'Backend {b} is not supported.')

    @staticmethod
    def set_and_return(tl_tensor, key, value):
        """
        for slices, make sure to use slice(), i.e., a : in traditional indexing can be replaced with slice(None)
        usage:

        a = T2.new([[1., 2.], [3., 4.]], Device.CPU, DType.Float64)
        a = T2.set(a, (slice(None), 1), 5.0)
        """
        b = T.get_backend()
        if b == 'numpy':
            tl_tensor[key] = value
        elif b == 'jax':
            tl_tensor = tl_tensor.at[key].set(value)
        elif b == 'pytorch':
            tl_tensor[key] = value
        else:
            raise ValueError(f'Backend {b} is not supported.')

        return tl_tensor
