from typing import TypeVar, Optional, Union, Tuple
import numpy as np
import scipy

try:
    import jax.numpy as jnp
    import jax.scipy as jsp
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
    def new_from_values(cls, row_major_values, backend: B = None) -> 'ApolloPyArray':
        if not backend:
            backend = ApolloPyArrayBackendNumpy()
        out = cls()
        out.array = backend.create_array(row_major_values)
        return out

    @classmethod
    def new(cls, array) -> 'ApolloPyArray':
        out = cls()
        out.array = array
        return out

    @staticmethod
    def zeros(shape, backend: B = None) -> 'ApolloPyArray':
        if not backend:
            backend = ApolloPyArrayBackendNumpy()
        a = np.zeros(shape)
        return ApolloPyArray.new_from_values(a, backend)

    @staticmethod
    def ones(shape, backend: B = None) -> 'ApolloPyArray':
        if not backend:
            backend = ApolloPyArrayBackendNumpy()
        a = np.ones(shape)
        return ApolloPyArray.new_from_values(a, backend)

    @staticmethod
    def diag(diag, backend: B = None) -> 'ApolloPyArray':
        if not backend:
            backend = ApolloPyArrayBackendNumpy()
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

    def __pow__(self, scalar) -> 'ApolloPyArray':
        return self.power(scalar)

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

    def is_scalar(self):
        return len(self.shape) == 0

    def diagonalize(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.diagonalize())

    def inv(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.inv())

    def pinv(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.pinv())

    def det(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.det())

    def trace(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.trace())

    def matrix_exp(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.matrix_exp())

    def svd(self, full_matrices: bool = True) -> 'SVDResult':
        return self.array.svd(full_matrices)

    def to_numpy_array(self):
        return self.array.to_numpy_array()

    def sin(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.sin())

    def cos(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.cos())

    def tan(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.tan())

    def arcsin(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.arcsin())

    def arccos(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.arccos())

    def arctan(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.arctan())

    def sinh(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.sinh())

    def cosh(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.cosh())

    def tanh(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.tanh())

    def exp(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.exp())

    def log(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.log())

    def log10(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.log10())

    def sqrt(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.sqrt())

    def abs(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.abs())

    def floor(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.floor())

    def ceil(self) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.ceil())

    def power(self, exponent) -> 'ApolloPyArray':
        return ApolloPyArray.new(self.array.power(exponent))

    def __getitem__(self, index):
        return ApolloPyArray.new(self.array.__getitem__(index))

    def __setitem__(self, index, value):
        self.array.__setitem__(index, value)

    def __str__(self):
        return self.array.__str__()

    def __repr__(self):
        return self.array.__repr__()

    def type(self):
        return type(self.array)

    def is_numpy(self):
        return self.type() == ApolloPyArrayNumpy

    def is_jax(self):
        return self.type() == ApolloPyArrayJAX

    def is_torch(self):
        return self.type() == ApolloPyArrayTorch


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

    def diagonalize(self) -> T:
        raise NotImplementedError('abstract base class')

    def inv(self) -> T:
        raise NotImplementedError('abstract base class')

    def pinv(self) -> T:
        raise NotImplementedError('abstract base class')

    def det(self) -> T:
        raise NotImplementedError('abstract base class')

    def trace(self) -> T:
        raise NotImplementedError('abstract base class')

    def matrix_exp(self) -> T:
        raise NotImplementedError('abstract base class')

    def svd(self, full_matrices: bool = False) -> 'SVDResult':
        raise NotImplementedError('abstract base class')

    def to_numpy_array(self) -> np.ndarray:
        raise NotImplementedError('abstract base class')

    def sin(self) -> T:
        raise NotImplementedError('abstract base class')

    def cos(self) -> T:
        raise NotImplementedError('abstract base class')

    def tan(self) -> T:
        raise NotImplementedError('abstract base class')

    def arcsin(self) -> T:
        raise NotImplementedError('abstract base class')

    def arccos(self) -> T:
        raise NotImplementedError('abstract base class')

    def arctan(self) -> T:
        raise NotImplementedError('abstract base class')

    def sinh(self) -> T:
        raise NotImplementedError('abstract base class')

    def cosh(self) -> T:
        raise NotImplementedError('abstract base class')

    def tanh(self) -> T:
        raise NotImplementedError('abstract base class')

    def exp(self) -> T:
        raise NotImplementedError('abstract base class')

    def log(self) -> T:
        raise NotImplementedError('abstract base class')

    def log10(self) -> T:
        raise NotImplementedError('abstract base class')

    def sqrt(self) -> T:
        raise NotImplementedError('abstract base class')

    def abs(self) -> T:
        raise NotImplementedError('abstract base class')

    def floor(self) -> T:
        raise NotImplementedError('abstract base class')

    def ceil(self) -> T:
        raise NotImplementedError('abstract base class')

    def power(self, exponent) -> T:
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

    def diagonalize(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.diag(self.array))

    def inv(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.linalg.inv(self.array))

    def pinv(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.linalg.pinv(self.array))

    def det(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.linalg.det(self.array))

    def trace(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.linalg.trace(self.array))

    def matrix_exp(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(scipy.linalg.expm(self.array))

    def svd(self, full_matrices: bool = False) -> 'SVDResult':
        U, S, VT = np.linalg.svd(self.array, full_matrices=full_matrices)
        U = ApolloPyArrayNumpy(U)
        S = ApolloPyArrayNumpy(S)
        VT = ApolloPyArrayNumpy(VT)
        return SVDResult(ApolloPyArray.new(U), ApolloPyArray.new(S), ApolloPyArray.new(VT))

    def to_numpy_array(self) -> np.ndarray:
        return self.array

    def sin(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.sin(self.array))

    def cos(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.cos(self.array))

    def tan(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.tan(self.array))

    def arcsin(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.arcsin(self.array))

    def arccos(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.arccos(self.array))

    def arctan(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.arctan(self.array))

    def sinh(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.sinh(self.array))

    def cosh(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.cosh(self.array))

    def tanh(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.tanh(self.array))

    def exp(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.exp(self.array))

    def log(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.log(self.array))

    def log10(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.log10(self.array))

    def sqrt(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.sqrt(self.array))

    def abs(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.abs(self.array))

    def floor(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.floor(self.array))

    def ceil(self) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.ceil(self.array))

    def power(self, exponent) -> 'ApolloPyArrayNumpy':
        return ApolloPyArrayNumpy(np.power(self.array, exponent))

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

        def diagonalize(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.diag(self.array))

        def inv(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.linalg.inv(self.array))

        def pinv(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.linalg.pinv(self.array))

        def det(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.linalg.det(self.array))

        def trace(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.linalg.trace(self.array))

        def matrix_exp(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jsp.linalg.expm(self.array))

        def svd(self, full_matrices: bool = False) -> 'SVDResult':
            U, S, VT = jnp.linalg.svd(self.array, full_matrices=full_matrices)
            U = ApolloPyArrayJAX(U)
            S = ApolloPyArrayJAX(S)
            VT = ApolloPyArrayJAX(VT)
            return SVDResult(ApolloPyArray.new(U), ApolloPyArray.new(S), ApolloPyArray.new(VT))

        def to_numpy_array(self) -> np.ndarray:
            return np.array(self.array)

        def sin(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.sin(self.array))

        def cos(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.cos(self.array))

        def tan(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.tan(self.array))

        def arcsin(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.arcsin(self.array))

        def arccos(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.arccos(self.array))

        def arctan(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.arctan(self.array))

        def sinh(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.sinh(self.array))

        def cosh(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.cosh(self.array))

        def tanh(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.tanh(self.array))

        def exp(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.exp(self.array))

        def log(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.log(self.array))

        def log10(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.log10(self.array))

        def sqrt(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.sqrt(self.array))

        def abs(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.abs(self.array))

        def floor(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.floor(self.array))

        def ceil(self) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.ceil(self.array))

        def power(self, exponent) -> 'ApolloPyArrayJAX':
            return ApolloPyArrayJAX(jnp.power(self.array, exponent))

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

        def diagonalize(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.diag(self.array))

        def inv(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.linalg.inv(self.array))

        def pinv(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.linalg.pinv(self.array))

        def det(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array.det())

        def trace(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(self.array.trace())

        def matrix_exp(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.linalg.matrix_exp(self.array))

        def svd(self, full_matrices: bool = False) -> 'SVDResult':
            U, S, VT = torch.linalg.svd(self.array, full_matrices=full_matrices)
            U = ApolloPyArrayTorch(U)
            S = ApolloPyArrayTorch(S)
            VT = ApolloPyArrayTorch(VT)
            return SVDResult(ApolloPyArray.new(U), ApolloPyArray.new(S), ApolloPyArray.new(VT))

        def to_numpy_array(self) -> np.ndarray:
            out = self.array.cpu().detach()

            return out.numpy()

        def sin(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.sin(self.array))

        def cos(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.cos(self.array))

        def tan(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.tan(self.array))

        def arcsin(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.arcsin(self.array))

        def arccos(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.arccos(self.array))

        def arctan(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.arctan(self.array))

        def sinh(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.sinh(self.array))

        def cosh(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.cosh(self.array))

        def tanh(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.tanh(self.array))

        def exp(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.exp(self.array))

        def log(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.log(self.array))

        def log10(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.log10(self.array))

        def sqrt(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.sqrt(self.array))

        def abs(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.abs(self.array))

        def floor(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.floor(self.array))

        def ceil(self) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.ceil(self.array))

        def power(self, exponent) -> 'ApolloPyArrayTorch':
            return ApolloPyArrayTorch(torch.pow(self.array, exponent))

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
