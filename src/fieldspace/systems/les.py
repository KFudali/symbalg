from scipy.sparse.linalg import LinearOperator, cg
import numpy as np

from discrete.core.bcs import BoundaryCondition, BCTool
from discrete.core.domain import Boundary
from algebra.symbolic import SymbolicExpression, SymbolicOperator, AffineOperator
from algebra.operator import Operator
from algebra.expression import Expression, CallableExpression


class LES:
    def __init__(
        self,
        bc_tool: BCTool[Operator, Boundary],
        lhs: Operator,
        rhs: Expression,
        bcs: list[BoundaryCondition[Boundary]],
    ):
        self._bc_tool = bc_tool
        if isinstance(lhs, AffineOperator):
            rhs -= lhs.expression
            lhs = lhs.operator
        self._rhs = rhs
        self._lhs = lhs
        self._bcs = bcs

    def _assemble(self) -> tuple[LinearOperator, np.ndarray]:
        rhs = self._rhs.eval()
        lhs = self._lhs
        if isinstance(lhs, SymbolicOperator):
            lhs = self._lhs.resolve()
        lhs = self._bc_tool.apply(self._bcs, lhs, rhs)
        out = np.zeros_like(rhs)

        def matvec(x: np.ndarray) -> np.ndarray:
            out[:] = 0.0
            lhs.apply(x.reshape(self._rhs.shape), out)
            return out.flatten()

        n = rhs.flatten().shape
        linop = LinearOperator(dtype=float, shape=(*n, *n), matvec=matvec)
        return linop, rhs

    def solve(self) -> SymbolicExpression:
        linop, rhs = self._assemble()
        def solve() -> np.ndarray:
            result, _ = cg(linop, rhs.flatten(), maxiter=1000, rtol=1e-9)
            result = result.reshape(rhs.shape)
            self._bc_tool.post_solve(self._bcs, result)
            return result

        return SymbolicExpression.wrap(CallableExpression(self._rhs.shape, solve))
