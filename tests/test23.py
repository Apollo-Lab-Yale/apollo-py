from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodReverseADPytorch, DerivativeMethodFD
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly import \
    BenchmarkFunction

f = BenchmarkFunction(2, 3, 100)
d = DerivativeMethodReverseADPytorch()
fe = FunctionEngine(f, d)

res = fe.derivative([1., 2.])
print(res)

d = DerivativeMethodFD()
fe = FunctionEngine(f, d)
res = fe.derivative([1., 2.])
print(res)

