import math

import numpy as np
import tensorly as tl
import torch
from tensorly import backend as T
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import ExtraBackend as T2, Device, DType

tl.set_backend('numpy')

a = T2.new([[1., 2.], [3., 4.]], Device.CPU, DType.Float64)

print(tl.is_tensor(T2.expm(a)))



