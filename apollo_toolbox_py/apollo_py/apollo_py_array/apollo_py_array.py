from typing import TypeVar, Optional, Union
import numpy as np


try:
    import jax.numpy as jnp
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

T = TypeVar('T', bound='ApolloPyArrayABC')
B = TypeVar('B', bound='ApolloPyArrayBackend')


class ApolloPyArray:
    def __init__(self, row_major_values, backend: B):
        self.array = backend.create_array(row_major_values)

    def mul(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.array @ other.array

    def __matmul__(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.array @ other.array

    def add(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.array + other.array

    def __add__(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.array + other.array

    def __getitem__(self, index):
        return self.array.__getitem__(index)

    def __setitem__(self, index, value):
        self.array.__setitem__(index, value)

    def __str__(self):
        return self.array.__str__()

    def __repr__(self):
        return self.array.__repr__()


class ApolloPyArrayBackend:
    def create_array(self, row_major_values) -> T:
        raise NotImplementedError('abstract base class')


class ApolloPyArrayBackendNumpy(ApolloPyArrayBackend):
    def create_array(self, row_major_values) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.array(row_major_values))


class ApolloPyArrayABC:
    def __init__(self, array):
        self.array = array

    def mul(self, other: T) -> T:
        raise NotImplementedError('abstract base class')

    def __matmul__(self, other: T) -> T:
        return self.mul(other)

    def add(self, other: T) -> T:
        raise NotImplementedError('abstract base class')

    def __add__(self, other: T) -> T:
        return self.add(other)

    def __getitem__(self, key):
        raise NotImplementedError('abstract base class')

    def __setitem__(self, key, value):
        raise NotImplementedError('abstract base class')

    def __repr__(self):
        return self.array.__repr__()

    def __str__(self):
        return self.array.__str__()


class ApolloPyArrayNumpy(ApolloPyArrayABC):
    def __init__(self, array):
        super().__init__(np.array(array))

    def mul(self, other: 'ApolloPyArrayNumpy') -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(self.array @ other.array)

    def add(self, other: 'ApolloPyArrayNumpy') -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(self.array + other.array)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value


if HAS_JAX:
    class ApolloPyArrayBackendJAX(ApolloPyArrayBackend):
        def __init__(self,
                     device: Optional[jax.Device] = None,
                     dtype: Optional[jnp.dtype] = None):
            """
            Initialize JAX backend with optional device and dtype specifications.

            Args:
                device: JAX device to place the array on (e.g., jax.devices()[0])
                dtype: Data type for the array (e.g., jnp.float32, jnp.float64)
            """
            self.device = device or jax.devices()[0]
            self.dtype = dtype

        def create_array(self, row_major_values) -> 'ApolloPyArrayJAX':
            # Convert to specified dtype or infer from input
            array = jnp.array(row_major_values, dtype=self.dtype)

            # Place on specified device
            with jax.default_device(self.device):
                return ApolloPyArrayJAX(array)


    class ApolloPyArrayJAX(ApolloPyArrayABC):
        def __init__(self, array):
            super().__init__(jnp.array(array))

        def mul(self, other: 'ApolloPyArrayJAX') -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(self.array @ other.array)

        def add(self, other: 'ApolloPyArrayJAX') -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(self.array + other.array)

        def __getitem__(self, key):
            return self.array[key]

        def __setitem__(self, key, value):
            self.array = self.array.at[key].set(value)

if HAS_PYTORCH:
    class ApolloPyArrayBackendTorch(ApolloPyArrayBackend):
        def __init__(self,
                     device: Optional[Union[str, torch.device]] = None,
                     dtype: Optional[torch.dtype] = None):
            """
            Initialize PyTorch backend with optional device and dtype specifications.

            Args:
                device: Device to place the tensor on (e.g., 'cuda', 'cpu', torch.device('cuda:0'))
                dtype: Data type for the tensor (e.g., torch.float32, torch.float64)
            """
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.dtype = dtype or torch.float64

        def create_array(self, row_major_values) -> 'ApolloPyArrayPyTorch':
            return ApolloPyArrayPyTorch(
                torch.tensor(
                    row_major_values,
                    device=self.device,
                    dtype=self.dtype
                )
            )


    class ApolloPyArrayPyTorch(ApolloPyArrayABC):
        def __init__(self, array):
            super().__init__(array)

        def mul(self, other: 'ApolloPyArrayPyTorch') -> 'ApolloPyArrayPyTorch':
            return ApolloPyArrayPyTorch(self.array @ other.array)

        def add(self, other: 'ApolloPyArrayPyTorch') -> 'ApolloPyArrayPyTorch':
            return ApolloPyArrayPyTorch(self.array + other.array)

        def __getitem__(self, key):
            return self.array[key]

        def __setitem__(self, key, value):
            if isinstance(value, torch.Tensor):
                self.array[key] = value
            else:
                self.array[key] = torch.tensor(value)

