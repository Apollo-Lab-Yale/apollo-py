from typing import TypeVar, Optional, Union, Tuple
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
    def __init__(self):
        self.array = None

    @classmethod
    def new_from_values(cls, row_major_values, backend: B) -> 'ApolloPyArray':
        out = cls()
        out.array = backend.create_array(row_major_values)
        return out

    @classmethod
    def new(cls, array) -> 'ApolloPyArray':
        out = cls()
        out.array = array
        return out

    @staticmethod
    def zeros(shape, backend: B) -> 'ApolloPyArray':
        a = np.zeros(shape)
        return ApolloPyArray.new_from_values(a, backend)

    @staticmethod
    def ones(shape, backend: B) -> 'ApolloPyArray':
        a = np.ones(shape)
        return ApolloPyArray.new_from_values(a, backend)

    @staticmethod
    def diag(diag, backend: B) -> 'ApolloPyArray':
        a = np.diag(diag)
        return ApolloPyArray.new_from_values(a, backend)

    def mul(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array @ other.array)

    def __matmul__(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.mul(other)

    def add(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array + other.array)

    def __add__(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.add(other)

    def sub(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array - other.array)

    def __sub__(self, other: 'ApolloPyArray') -> 'ApolloPyArray':
        return self.sub(other)

    def scalar_mul(self, scalar) -> 'ApolloPyArray':
        return ApolloPyArray.new(scalar * self.array)

    def __mul__(self, scalar) -> 'ApolloPyArray':
        return self.scalar_mul(scalar)

    def __rmul__(self, scalar) -> 'ApolloPyArray':
        return self.scalar_mul(scalar)

    def scalar_div(self, scalar) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array / scalar)

    def __truediv__(self, scalar) -> 'ApolloPyArray':
        return self.scalar_div(scalar)

    def transpose(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.transpose())

    @property
    def T(self):
        return self.transpose()

    def resize(self, new_shape: Tuple[int, ...]) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.resize(new_shape))

    @property
    def shape(self):
        return self.array.array.shape

    def svd(self, full_matrices: bool = True) -> 'SVDResult':
        return self.array.svd(full_matrices)

    def __getitem__(self, index):
        return ApolloPyArray.new(self.array.__getitem__(index))

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

    def sub(self, other: T) -> T:
        raise NotImplementedError('abstract base class')

    def __sub__(self, other: T) -> T:
        return self.sub(other)

    def scalar_mul(self, scalar) -> T:
        raise NotImplementedError('abstract base class')

    def __mul__(self, scalar) -> T:
        return self.scalar_mul(scalar)

    def __rmul__(self, scalar) -> T:
        return self.scalar_mul(scalar)

    def scalar_div(self, scalar) -> T:
        raise NotImplementedError('abstract base class')

    def __truediv__(self, scalar) -> T:
        return self.scalar_div(scalar)

    def transpose(self) -> T:
        raise NotImplementedError('abstract base class')

    @property
    def T(self):
        return self.transpose()

    def resize(self, new_shape: Tuple[int, ...]) -> T:
        raise NotImplementedError('abstract base class')

    def svd(self, full_matrices: bool = False) -> 'SVDResult':
        raise NotImplementedError('abstract base class')

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

    def sub(self, other: 'ApolloPyArrayNumpy') -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(self.array - other.array)

    def scalar_mul(self, scalar) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(self.array * scalar)

    def scalar_div(self, scalar) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(self.array / scalar)

    def transpose(self) -> T:
        return ApolloPyArrayNumpy(self.array.transpose())

    def resize(self, new_shape: Tuple[int, ...]) -> 'ApolloPyArrayNumpy':
        """
        Resize the NumPy array to a new shape.

        Args:
            new_shape: A tuple representing the new shape of the array

        Returns:
            A new NumPy-backed ApolloPyArray with the specified shape
        """
        return ApolloPyArrayNumpy(np.resize(self.array, new_shape))

    def svd(self, full_matrices: bool = False) -> 'SVDResult':
        U, S, VT = np.linalg.svd(self.array, full_matrices=full_matrices)
        U = ApolloPyArrayNumpy(U)
        S = ApolloPyArrayNumpy(S)
        VT = ApolloPyArrayNumpy(VT)
        return SVDResult(ApolloPyArray.new(U), ApolloPyArray.new(S), ApolloPyArray.new(VT))

    def __getitem__(self, key):
        return ApolloPyArrayNumpy(self.array[key])

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

        def sub(self, other: 'ApolloPyArrayJAX') -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(self.array - other.array)

        def scalar_mul(self, scalar) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(self.array * scalar)

        def scalar_div(self, scalar) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(self.array / scalar)

        def transpose(self) -> T:
            return ApolloPyArrayJAX(self.array.transpose())

        def resize(self, new_shape: Tuple[int, ...]) -> 'ApolloPyArrayJAX':
            """
            Resize the JAX array to a new shape.

            Args:
                new_shape: A tuple representing the new shape of the array

            Returns:
                A new JAX-backed ApolloPyArray with the specified shape
            """
            return ApolloPyArrayJAX(jnp.resize(self.array, new_shape))

        def svd(self, full_matrices: bool = False) -> 'SVDResult':
            U, S, VT = jnp.linalg.svd(self.array, full_matrices=full_matrices)
            U = ApolloPyArrayJAX(U)
            S = ApolloPyArrayJAX(S)
            VT = ApolloPyArrayJAX(VT)
            return SVDResult(ApolloPyArray.new(U), ApolloPyArray.new(S), ApolloPyArray.new(VT))

        def __getitem__(self, key):
            return ApolloPyArrayJAX(self.array[key])

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

        def create_array(self, row_major_values) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(
                torch.tensor(
                    row_major_values,
                    device=self.device,
                    dtype=self.dtype
                )
            )


    class ApolloPyArrayTorch(ApolloPyArrayABC):
        def __init__(self, array):
            super().__init__(array)

        def mul(self, other: 'ApolloPyArrayTorch') -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array @ other.array)

        def add(self, other: 'ApolloPyArrayTorch') -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array + other.array)

        def sub(self, other: 'ApolloPyArrayTorch') -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array - other.array)

        def scalar_mul(self, scalar) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array * scalar)

        def scalar_div(self, scalar) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array / scalar)

        def transpose(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array.T)

        def resize(self, new_shape: Tuple[int, ...]) -> 'ApolloPyArrayTorch':
            """
            Resize the PyTorch tensor to a new shape.

            Args:
                new_shape: A tuple representing the new shape of the array

            Returns:
                A new PyTorch-backed ApolloPyArray with the specified shape
            """
            return ApolloPyArrayTorch(self.array.reshape(new_shape))

        def svd(self, full_matrices: bool = False) -> 'SVDResult':
            U, S, VT = torch.linalg.svd(self.array, full_matrices=full_matrices)
            U = ApolloPyArrayTorch(U)
            S = ApolloPyArrayTorch(S)
            VT = ApolloPyArrayTorch(VT)
            return SVDResult(ApolloPyArray.new(U), ApolloPyArray.new(S), ApolloPyArray.new(VT))

        def __getitem__(self, key):
            return ApolloPyArrayTorch(self.array[key])

        def __setitem__(self, key, value):
            if isinstance(value, torch.Tensor):
                self.array[key] = value
            else:
                self.array[key] = torch.tensor(value)


class SVDResult:
    def __init__(self, U: ApolloPyArray, singular_vals: ApolloPyArray, VT: ApolloPyArray):
        self.U = U
        self.singular_vals = singular_vals
        self.VT = VT
