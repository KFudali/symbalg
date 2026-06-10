from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
from typing import Optional

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from algebra.operator import Operator
from algebra.systems.systems import LinearSystem


class SolverError(Exception):
    pass


class LinearSolver(ABC):
    @abstractmethod
    def solve(self, system: LinearSystem) -> np.ndarray:
        pass


class CGSolver(LinearSolver):
    def __init__(
        self,
        *,
        maxiter: int = 1000,
        rtol: float = 1e-9,
        atol: float = 0.0,
        precond: Optional[Operator] = None,
    ):
        self._maxiter = maxiter
        self._rtol = rtol
        self._atol = atol
        self._precond = precond

    def solve(self, system: LinearSystem) -> np.ndarray:
        shape = system.rhs.shape
        linop = self._assemble_operator(system.lhs, shape)
        precond = self._assemble_precond(shape)

        result, info = cg(
            linop,
            system.rhs.ravel(),
            maxiter=self._maxiter,
            rtol=self._rtol,
            atol=self._atol,
            M=precond,
        )

        if info == 0:
            return result.reshape(shape)
        if info > 0:
            warnings.warn(
                f"CG did not converge after {self._maxiter} iterations",
                RuntimeWarning,
            )
            return result.reshape(shape)
        raise SolverError(f"cg returned error code {info}")

    @staticmethod
    def _assemble_operator(
        operator: Operator, shape: tuple[int, ...]
    ) -> LinearOperator:
        n = int(np.prod(shape))
        out = np.zeros(shape, dtype=float)

        def matvec(x: np.ndarray) -> np.ndarray:
            out[:] = 0.0
            operator.apply(x.reshape(shape), out)
            return out.ravel()

        return LinearOperator(dtype=float, shape=(n, n), matvec=matvec)

    def _assemble_precond(self, shape: tuple[int, ...]) -> Optional[LinearOperator]:
        if self._precond is None:
            return None
        return self._assemble_operator(self._precond, shape)
