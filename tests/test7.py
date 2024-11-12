from apollo_toolbox_py.apollo_py.apollo_py_derivatives.derivative_engine_evaluator import DerivativeEngineEvaluator
from apollo_toolbox_py.apollo_py.apollo_py_derivatives.test_functions import MatrixMultiply

m = 50
n = 50

f = MatrixMultiply.new_random(m, n, 10)
e = DerivativeEngineEvaluator(f, n, m, True, 1000, 0.1)
e.evaluate()

print(e.wasp_engine_errors)
print(e.wasp_num_f_calls)
