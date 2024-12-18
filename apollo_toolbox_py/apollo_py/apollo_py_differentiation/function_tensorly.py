from abc import ABC, abstractmethod

from apollo_toolbox_py.apollo_py_tensorly.apollo_py_tensorly_linalg.vectors import V
import tensorly as tl


class FunctionTensorly(ABC):
    def call(self, x: V) -> V:
        assert len(x.array) == self.input_dim()
        out = self.call_raw(x)
        assert len(out.array) == self.output_dim()
        return out

    @abstractmethod
    def call_raw(self, x: V) -> V:
        pass

    @abstractmethod
    def input_dim(self):
        pass

    @abstractmethod
    def output_dim(self):
        pass


class TestFunction(FunctionTensorly):
    def call_raw(self, x: V) -> V:
        return V([tl.sin(x[0])])

    def input_dim(self):
        return 1

    def output_dim(self):
        return 1
