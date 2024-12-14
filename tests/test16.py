import math

import numpy as np
import tensorly as tl
import torch
from tensorly import backend as T
from apollo_toolbox_py.apollo_py.extra_tensorly_functions import ExtraTensorlyFunctions as etf

tl.set_backend('numpy')

a = tl.tensor([[1., 2], [3, 4]])
# U, S, V = T.svd(a)
# print(isinstance(a, np.ndarray))

print(etf.det(a))



