from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.isometries import Isometry3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.h1 import LieGroupH1, LieAlgH1
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.lie.so3 import LieAlgSO3
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_spatial.quaternions import UnitQuaternion, Quaternion
from apollo_toolbox_py.apollo_py_numpy.apollo_py_numpy_linalg.vectors import V3, V6
import numpy as np


class LieGroupISE3q(Isometry3):
    @classmethod
    def identity(cls) -> 'LieGroupISE3q':
        return LieGroupISE3q(UnitQuaternion.new_unchecked([1, 0, 0, 0]), V3([0, 0, 0]))

    def ln(self) -> 'LieAlgISE3q':
        a_quat = LieGroupH1(self.rotation.array).ln()
        u = a_quat.vee()
        a_mat: LieAlgSO3 = u.to_lie_alg_so3()
        beta = np.linalg.norm(u.array)

        if abs(beta) < 0.00001:
            pp = 0.5 - ((beta ** 2.0) / 24.0) + ((beta ** 4.0) / 720.0)
            qq = (1.0 / 6.0) - ((beta ** 2.0) / 120.0) + ((beta ** 4.0) / 5040.0)
        else:
            pp = (1.0 - np.cos(beta)) / (beta ** 2.0)
            qq = (beta - np.sin(beta)) / (beta ** 3.0)

        c_mat = np.identity(3) + (pp * a_mat.array) + qq * (a_mat.array @ a_mat.array)
        c_inv = np.linalg.inv(c_mat)

        b = V3(c_inv @ self.translation.array)

        return LieAlgISE3q(a_quat, b)

    def __repr__(self) -> str:
        return f"LieGroupISE3q(\n  rotation: {np.array2string(self.rotation.array)},\n  ----- \n  translation: {np.array2string(self.translation.array)}\n)"

    def __str__(self) -> str:
        return f"LieGroupISE3q(\n  rotation: {np.array2string(self.rotation.array)},\n  ----- \n  translation: {np.array2string(self.translation.array)}\n)"


class LieAlgISE3q:
    def __init__(self, quaternion: Quaternion, vector: V3):
        self.quaternion = quaternion
        self.vector = vector

    @classmethod
    def from_euclidean_space_element(cls, e: V6) -> 'LieAlgISE3q':
        u = V3([e[0], e[1], e[2]])
        q = u.to_lie_alg_h1()
        v = V3([e[3], e[4], e[5]])
        return LieAlgISE3q(q, v)

    def exp(self) -> 'LieGroupISE3q':
        u = LieAlgH1(self.quaternion.array).vee()
        a_mat: LieAlgSO3 = u.to_lie_alg_so3()

        beta = np.linalg.norm(u.array)

        if abs(beta) < 0.00001:
            pp = 0.5 - ((beta ** 2.0) / 24.0) + ((beta ** 4.0) / 720.0)
            qq = (1.0 / 6.0) - ((beta ** 2.0) / 120.0) + ((beta ** 4.0) / 5040.0)
        else:
            pp = (1.0 - np.cos(beta)) / (beta ** 2.0)
            qq = (beta - np.sin(beta)) / (beta ** 3.0)

        c_mat = np.identity(3) + pp * a_mat.array + qq * (a_mat.array @ a_mat.array)
        t = V3(c_mat @ self.vector.array)
        q = LieAlgH1(self.quaternion.array).exp()

        return LieGroupISE3q(UnitQuaternion(q.array), t)

    def vee(self) -> 'V6':
        u = LieAlgH1(self.quaternion.array).vee()
        v = self.vector

        return V6([u[0], u[1], u[2], v[0], v[1], v[2]])

    def __repr__(self) -> str:
        return f"LieAlgISE3q(\n  quaternion: {np.array2string(self.quaternion.array)},\n  ----- \n  vector: {np.array2string(self.vector.array)}\n)"

    def __str__(self) -> str:
        return f"LieAlgISE3q(\n  quaternion: {np.array2string(self.quaternion.array)},\n  ----- \n  vector: {np.array2string(self.vector.array)}\n)"
