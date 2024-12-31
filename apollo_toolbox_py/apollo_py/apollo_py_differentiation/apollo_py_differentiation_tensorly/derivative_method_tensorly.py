from abc import ABC, abstractmethod
from typing import List

import jax
import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    FunctionTensorly
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend, DType, Device
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.matrices import M
from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2


class DerivativeMethodTensorly(ABC):
    @abstractmethod
    def allowable_backends(self) -> List[Backend]:
        pass

    @abstractmethod
    def default_backend(self) -> Backend:
        pass

    @abstractmethod
    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        pass

    def derivative(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        assert x.shape == (f.input_dim(),)
        dfdx = self.derivative_raw(f, x)
        assert dfdx.shape == (f.output_dim(), f.input_dim()), 'shape is {}'.format(dfdx.shape)
        return dfdx


class DerivativeMethodFD(DerivativeMethodTensorly):
    def __init__(self, epsilon=0.000001):
        self.epsilon = epsilon

    def allowable_backends(self) -> List[Backend]:
        return [Backend.Numpy, Backend.JAX, Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.Numpy

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        fx = f.call(x)
        dfdx = tl.zeros((f.output_dim(), f.input_dim()), device=getattr(x, 'device', None), dtype=x.dtype)

        for i in range(f.input_dim()):
            delta_x = tl.zeros(f.input_dim(), device=getattr(x, 'device', None), dtype=x.dtype)
            delta_x = T2.set_and_return(delta_x, i, self.epsilon)
            x_delta_x = x + delta_x
            fh = f.call(x_delta_x)
            col = (fh - fx) / self.epsilon
            dfdx = T2.set_and_return(dfdx, (slice(None), i), col)

        return dfdx


class DerivativeMethodReverseADJax(DerivativeMethodTensorly):
    def __init__(self):
        self.jac_fn = None

    def allowable_backends(self) -> List[Backend]:
        return [Backend.JAX]

    def default_backend(self) -> Backend:
        return Backend.JAX

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        if not self.jac_fn:
            self.jac_fn = jax.jacrev(f.call)

        return self.jac_fn(x)


class DerivativeMethodForwardADJax(DerivativeMethodTensorly):
    def __init__(self):
        self.jac_fn = None

    def allowable_backends(self) -> List[Backend]:
        return [Backend.JAX]

    def default_backend(self) -> Backend:
        return Backend.JAX

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        if not self.jac_fn:
            self.jac_fn = jax.jacrev(f.call)

        return self.jac_fn(x)


class DerivativeMethodReverseADPytorch(DerivativeMethodTensorly):
    def allowable_backends(self) -> List[Backend]:
        return [Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.PyTorch

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        dfdx = tl.zeros(f.output_dim(), f.input_dim(), device=getattr(x, 'device', None), dtype=x.dtype)

        x.requires_grad = True
        fx = f.call(x)
        for i in range(f.output_dim()):
            if x.grad is not None:
                x.grad.zero_()

            fx[i].backward(retain_graph=True)
            col = x.grad.clone()
            dfdx[i, :] = col

        return dfdx


class DerivativeMethodWASP(DerivativeMethodTensorly):
    def __init__(self, n: int, m: int, backend: Backend, orthonormal: bool = True, d_ell=0.3, d_theta=0.3,
                 device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        tl.set_backend(backend.to_string())
        self.cache = WASPCache(n, m, orthonormal, device, dtype)
        self.num_f_calls = 0
        self.d_theta = d_theta
        self.d_ell = d_ell

    def allowable_backends(self) -> List[Backend]:
        return [Backend.Numpy, Backend.JAX, Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.Numpy

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        self.num_f_calls = 0
        f_k = f.call(x)
        self.num_f_calls += 1
        epsilon = 0.000001

        cache = self.cache

        while True:
            i = self.cache.i

            delta_x_i = cache.delta_x[:, i]

            x_k_plus_delta_x_i = x + epsilon * delta_x_i
            f_k_plus_delta_x_i = f.call(x_k_plus_delta_x_i)
            self.num_f_calls += 1
            delta_f_i = (f_k_plus_delta_x_i - f_k) / epsilon
            delta_f_i_hat = cache.delta_f_t[i, :]
            return_result = close_enough(delta_f_i, delta_f_i_hat, self.d_theta, self.d_ell)

            # cache.delta_f_t[i, :] = delta_f_i
            cache.delta_f_t = T2.set_and_return(cache.delta_f_t, (i, slice(None)), delta_f_i)
            c_1_mat = cache.c_1[i]
            c_2_mat = cache.c_2[i]
            delta_f_t = cache.delta_f_t
            delta_f_i = tl.reshape(delta_f_i, (-1, 1))

            d_t_star = c_1_mat @ delta_f_t + c_2_mat @ delta_f_i.T
            d_star = d_t_star.T

            tmp = d_star @ cache.delta_x
            cache.delta_f_t = tmp.T

            new_i = i + 1
            if new_i >= len(x):
                new_i = 0
            cache.i = new_i

            if return_result:
                return d_star


class WASPCache:
    def __init__(self, n: int, m: int, orthonormal_delta_x: bool = True, device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        self.i = 0
        self.delta_f_t = T2.new(np.eye(n, m), device=device, dtype=dtype)
        delta_x = get_tangent_matrix(n, orthonormal_delta_x, device, dtype)
        self.c_1 = []
        self.c_2 = []

        a_mat = 2.0 * delta_x @ delta_x.T
        a_inv_mat = T2.inv(a_mat)
        eye = T2.new(np.eye(n, n), device=device, dtype=dtype)

        for i in range(n):
            delta_x_i = delta_x[:, i:i + 1]
            s_i = delta_x_i.T @ a_inv_mat @ delta_x_i
            s_i_inv = 1.0 / s_i
            c_1_mat = a_inv_mat @ (eye - s_i_inv * delta_x_i @ delta_x_i.T @ a_inv_mat) @ (2.0 * delta_x)
            c_2_mat = s_i_inv * a_inv_mat @ delta_x_i
            self.c_1.append(c_1_mat)
            self.c_2.append(c_2_mat)

        self.delta_x = delta_x


class DerivativeMethodWASP2(DerivativeMethodTensorly):
    def __init__(self, n: int, m: int, backend: Backend, alpha: float = 0.98, orthonormal: bool = True, d_ell=0.3,
                 d_theta=0.3,
                 max_f_calls: int = 9999999,
                 device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        tl.set_backend(backend.to_string())
        self.cache = WASPCache2(n, m, alpha, orthonormal, device, dtype)
        self.num_f_calls = 0
        self.d_theta = d_theta
        self.d_ell = d_ell
        self.max_f_calls = max_f_calls

    def allowable_backends(self) -> List[Backend]:
        return [Backend.Numpy, Backend.JAX, Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.Numpy

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        self.num_f_calls = 0
        f_k = f.call(x)
        self.num_f_calls += 1
        epsilon = 0.000001

        cache = self.cache

        while True:
            i = self.cache.i

            delta_x_i = cache.delta_x[:, i]

            x_k_plus_delta_x_i = x + epsilon * delta_x_i
            f_k_plus_delta_x_i = f.call(x_k_plus_delta_x_i)
            self.num_f_calls += 1
            delta_f_i = (f_k_plus_delta_x_i - f_k) / epsilon
            # delta_f_i_hat = cache.delta_f_t[i, :]
            # tmp = tl.reshape(delta_x_i, (-1, 1))
            delta_f_i_hat = self.cache.curr_d @ delta_x_i
            return_result = close_enough(delta_f_i, delta_f_i_hat, self.d_theta, self.d_ell)

            # cache.delta_f_t[i, :] = delta_f_i
            cache.delta_f_t = T2.set_and_return(cache.delta_f_t, (i, slice(None)), delta_f_i)
            c_1_mat = cache.c_1[i]
            c_2_mat = cache.c_2[i]
            delta_f_t = cache.delta_f_t
            delta_f_i = tl.reshape(delta_f_i, (-1, 1))

            d_t_star = c_1_mat @ delta_f_t + c_2_mat @ delta_f_i.T
            d_star = d_t_star.T
            self.cache.curr_d = d_star

            new_i = i + 1
            if new_i >= len(x):
                new_i = 0
            cache.i = new_i

            if return_result or self.num_f_calls >= self.max_f_calls:
                return d_star


class WASPCache2:
    def __init__(self, n: int, m: int, alpha: float = 0.98, orthonormal_delta_x: bool = True,
                 device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        self.i = 0
        self.curr_d = T2.new(np.zeros((m, n)), device=device, dtype=dtype)
        self.delta_f_t = T2.new(np.eye(n, m), device=device, dtype=dtype)
        delta_x = get_tangent_matrix(n, orthonormal_delta_x, device, dtype)
        self.c_1 = []
        self.c_2 = []

        eye = T2.new(np.eye(n, n), device=device, dtype=dtype)

        for i in range(n):
            delta_x_i = delta_x[:, i:i + 1]
            w_i = T2.new(np.zeros((n, n)), device=device, dtype=dtype)
            for j in range(n):
                exponent = float(math_mod(i - j, n)) / float(n - 1)
                w_i = T2.set_and_return(w_i, (j, j), alpha * (1.0 - alpha) ** exponent)
            w_i_2 = w_i @ w_i

            a_i = 2.0 * delta_x @ w_i_2 @ delta_x.T
            a_i_inv = T2.inv(a_i)

            s_i = delta_x_i.T @ a_i_inv @ delta_x_i
            s_i_inv = 1.0 / s_i
            c_1_mat = a_i_inv @ (eye - s_i_inv * delta_x_i @ delta_x_i.T @ a_i_inv) @ (2.0 * delta_x @ w_i_2)
            c_2_mat = s_i_inv * a_i_inv @ delta_x_i
            self.c_1.append(c_1_mat)
            self.c_2.append(c_2_mat)

        self.delta_x = delta_x


class DerivativeMethodWASP3(DerivativeMethodTensorly):
    def __init__(self, n: int, m: int, backend=Backend, h: int = None, device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        if h is None:
            h = n

        assert h > 0

        tl.set_backend(backend.to_string())

        self.n = n
        self.m = m
        self.h = h
        self.device = device
        self.dtype = dtype

        self.first_pass = True

        self.delta_x = T2.new(np.zeros((m, n)), device=device, dtype=dtype)
        self.w = T2.new(np.eye(h, h), device=device, dtype=dtype)
        self.delta_f_t = T2.new(np.zeros((n, m)), device=device, dtype=dtype)
        self.eye = T2.new(np.eye(n, n), device=device, dtype=dtype)

        self.x_inputs = T2.new(np.zeros((n, h)), device=device, dtype=dtype)
        self.f_outputs = T2.new(np.zeros((m, h)), device=device, dtype=dtype)

    def allowable_backends(self) -> List[Backend]:
        return [Backend.Numpy, Backend.JAX, Backend.PyTorch]

    def default_backend(self) -> Backend:
        return Backend.Numpy

    def derivative_raw(self, f: FunctionTensorly, x: tl.tensor) -> tl.tensor:
        if self.first_pass:
            for i in range(self.h):
                r = T2.new(np.random.uniform(-0.00001, 0.00001, (self.n,)), self.device, self.dtype)
                rpx = x + r
                frpx = f.call(rpx)
                self.x_inputs = T2.set_and_return(self.x_inputs, (slice(None), i), rpx)
                self.f_outputs = T2.set_and_return(self.f_outputs, (slice(None), i), frpx)
            self.first_pass = False

        x = x + T2.new(np.random.uniform(-0.000001, 0.000001), self.device, self.dtype)
        x_as_mat = tl.reshape(x, (-1, 1))
        fx = f.call(x)
        fx_as_mat = tl.reshape(fx, (-1, 1))

        distances = tl.norm(self.x_inputs - x_as_mat, axis=0)
        w_diag_squared = (0.01 / distances) ** 2

        W2 = tl.diag(w_diag_squared)
        Delta_x = self.x_inputs - x_as_mat
        Delta_f_hat = self.f_outputs - fx_as_mat

        max_distance_idx = tl.argmax(distances)
        min_distance_idx = tl.argmin(distances)

        delta_x_i = Delta_x[:, min_distance_idx:min_distance_idx + 1]
        delta_f_i = Delta_f_hat[:, min_distance_idx:min_distance_idx + 1]

        eye = self.eye
        A = 2.0 * Delta_x @ W2 @ Delta_x.T
        A_inv = T2.inv(A)
        s = (delta_x_i.T @ A_inv @ delta_x_i)[0, 0]
        s_inv = 1.0 / s

        DT = A_inv @ (eye - s_inv * delta_x_i @ delta_x_i.T @ A_inv) @ (2.0 * Delta_x @ W2 @ Delta_f_hat.T) + s_inv * A_inv @ delta_x_i @ delta_f_i.T

        self.x_inputs = T2.set_and_return(self.x_inputs, (slice(None), max_distance_idx), x)
        self.f_outputs = T2.set_and_return(self.f_outputs, (slice(None), max_distance_idx), fx)

        return DT.T


'''
class WASPCache3:
    def __init__(self, n: int, m: int, h: int = None, device: Device = Device.CPU,
                 dtype: DType = DType.Float64):
        if h is None:
            h = n

        assert h > 0

        self.n = n
        self.m = m
        self.h = h
        self.device = device
        self.dtype = dtype

        self.first_pass = True

        self.delta_x = T2.new(np.zeros((m, n)), device=device, dtype=dtype)
        self.w = T2.new(np.eye(h, h), device=device, dtype=dtype)
        self.delta_f_t = T2.new(np.zeros((n, m)), device=device, dtype=dtype)
        self.eye = T2.new(np.eye(n, n), device=device, dtype=dtype)

        self.x_inputs = []
        self.f_outputs = []
'''


def math_mod(a: int, b: int) -> int:
    return ((a % b) + b) % b


def get_tangent_matrix(n: int, orthonormal: bool, device: Device = Device.CPU,
                       dtype: DType = DType.Float64) -> tl.tensor:
    t = np.random.uniform(-1, 1, (n, n))
    t = T2.new(t, device=device, dtype=dtype)
    if orthonormal:
        U, S, VT = tl.svd(t, full_matrices=True)
        delta_x = U @ VT
        return delta_x
    else:
        return t


def close_enough(a: tl.tensor, b: tl.tensor, d_theta: float, d_ell: float):
    a_n = tl.norm(a)
    b_n = tl.norm(b)

    if a_n == 0.0 or b_n == 0.0:
        return False

    tmp = tl.abs((tl.dot(a, b) / (a_n * b_n)) - 1.0)
    if tmp > d_theta:
        return False

    if not b_n == 0.0:
        tmp1 = tl.abs((a_n / b_n) - 1.0)
    else:
        tmp1 = 10000000.0
    if not a_n == 0.0:
        tmp2 = tl.abs((b_n / a_n) - 1.0)
    else:
        tmp2 = 10000000.0

    if min(tmp1, tmp2) > d_ell:
        return False

    return True
