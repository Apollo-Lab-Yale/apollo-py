import numpy as np

from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    close_enough, get_tangent_matrix, WASPCache
import tensorly as tl
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2

tl.set_backend('pytorch')

res = WASPCache(3, 3, True)
