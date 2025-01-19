from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodReverseADPytorch
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction, BenchmarkFunction2

f = BenchmarkFunction2(2, 3, 10000)
d = DerivativeMethodReverseADPytorch()
fe = FunctionEngine(f, d)

print(fe.derivative([10000., 2.]))
print(fe.derivative([10000., 2.8]))

