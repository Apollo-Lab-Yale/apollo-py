from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodWASP2
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend

f = BenchmarkFunction(3, 4, 100)
d = DerivativeMethodWASP2(3, 4, Backend.Numpy, orthonormal=False)
fe = FunctionEngine(f, d)

res = fe.derivative([1., 2., 3.])
print(res)
print(fe.d.num_f_calls)

res = fe.derivative([1., 2., 3.])
print(res)
print(fe.d.num_f_calls)

res = fe.derivative([1., 2., 3.])
print(res)
print(fe.d.num_f_calls)

res = fe.derivative([1., 2., 3.])
print(res)
print(fe.d.num_f_calls)

