import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from algebra.expression import Expression, CallableExpression
from algebra.symbolic import SymbolicExpression, SymbolicOperator, AffineOperator

from .bcs import BoundaryCondition, BCTool


class LES():
    def __init__(
        self,
        lhs: AffineOperator,
        rhs: Expression,
        bc_tool: BCTool
    ):
        self._lhs = lhs
        self._rhs = rhs
        self._bc_tool = bc_tool
        self._bcs = set[BoundaryCondition]()

    def solve(self) -> Expression:
        linop, rhs = self._assemble()
        def solve() -> np.ndarray:
            x, _ = cg(linop, rhs, maxiter=1000, rtol=1e-9)
            x = x.rjeshape(self._lhs.output_shape)
            for bc in self._bcs:
                self._bc_tool.post_solve(bc, x)
            return SymbolicExpression(CallableExpression(solve, rhs.shape))

    def _assemble(self) -> tuple[LinearOperator, np.ndarray]:
        lhs = self._lhs.operator.copy()
        for bc in self._bcs:
            self._bc_tool.apply(bc, lhs)
        rhs = self._rhs.eval() - lhs.expression.eval()
        n = rhs.flatten().shape
        if isinstance(lhs, SymbolicOperator):
            lhs = lhs.fold()
        def matvec(x: np.ndarray):
            out = np.zeros_like(x)
            lhs.apply(x.reshape(lhs.input_shape), out.reshape(lhs.output_shape))
            return out

        linop = LinearOperator(dtype=float, shape=(*n, *n), matvec = matvec)
        return linop, rhs


    def add_bcs(self, bcs: list[BoundaryCondition]):
        for bc in bcs:
            self._bcs.add(bc)
