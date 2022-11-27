from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from transforms3d import affines


class Transform3D:
    def __init__(self, Tmat: npt.NDArray):
        if (4, 4) != Tmat.shape:
            raise ValueError("4x4 homogeneous transformation matrix expected")
        self._Tmat = Tmat

    @classmethod
    def from_rmat_t(
        cls, rmat: Optional[npt.NDArray] = None, t: Optional[npt.NDArray] = None
    ) -> Transform3D:
        if rmat is None:
            rmat = np.eye(3, dtype=float)
        if rmat.shape != (3, 3):
            raise ValueError("3x3 matrix expected")
        if t is None:
            t = np.zeros((3,), dtype=float)
        t = t.reshape((3,))
        if t.size != 3:
            raise ValueError("3-element translation vector expected")

        return cls(affines.compose(t, rmat, np.ones(3)))

    @classmethod
    def identity(cls) -> Transform3D:
        return cls.from_rmat_t(np.eye(3, dtype=float), np.zeros((3,), dtype=float))

    @property
    def Tmat(self):
        return self._Tmat

    @property
    def t(self):
        return self._Tmat[:3, 3]

    @t.setter
    def t(self, value: npt.NDArray[float]):
        self._Tmat[:3, 3] = value

    @property
    def Rmat(self):
        return self._Tmat[:3, :3]

    def inv(self):
        return self.__class__(np.linalg.inv(self.Tmat))

    def __mul__(self, other: Transform3D) -> Transform3D:
        if isinstance(other, Transform3D):
            return self.__class__(self.Tmat @ other.Tmat)
        raise TypeError(
            f"Multiplication is only supported between {self.__class__} objects."
        )

    def __matmul__(self, other: Transform3D) -> Transform3D:
        return self * other

    def __str__(self) -> str:
        return f"Homogeneous transformation(\n{self.Tmat})"
