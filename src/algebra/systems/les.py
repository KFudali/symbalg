import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from algebra.exceptions import ShapeMismatchError
from algebra.core.expression import Expression, CallableExpression
from algebra.core.operator import SpaceOperator

from .bcs import BoundaryCondition, BCTool

class LES():
    def __init__(
        self,
        lhs: SpaceOperator,
        rhs: Expression,
        bc_tool: BCTool
    ):
        if lhs.input_shape != lhs.output_shape:
            raise ShapeMismatchError((
                "LES can only be created using Operator of the same input and ",
                f"output shapes. Got input_shape: {lhs.input_shape} , ",
                f"output_shape: {lhs.output_shape}."
            ))
        if lhs.output_shape != rhs.output_shape:
            raise ShapeMismatchError((
                f"rhs shape: f{rhs.output_shape} has to match lhs shape ",
                f"{lhs.output_shape}."
            ))
        self._lhs = lhs
        self._rhs = rhs
        self._bc_tool = bc_tool
        self._bcs = set[BoundaryCondition]()

    def solve(self) -> Expression:
        linop, rhs = self._assemble()
        def solve() -> np.ndarray:
            result, _ = cg(linop, rhs, maxiter=1000, rtol=1e-9)
            result = result.reshape(self._lhs.output_shape)
            for bc in self._bcs:
                for comp in range(self._lhs.output_components):
                    self._bc_tool.post_solve(bc, result[comp])
            return result
        return CallableExpression(solve, self._lhs.output_shape)

    def _assemble(self) -> tuple[LinearOperator, np.ndarray]:
        rhs = self._rhs.eval()
        lhs = self._lhs.copy()
        self._bc_tool.apply(list(self._bcs), lhs, rhs)
        n = rhs.flatten().shape
        def matvec(x: np.ndarray):
            out = np.zeros_like(x)
            lhs.apply(x.reshape(lhs.input_shape), out.reshape(lhs.output_shape))
            return out
        linop = LinearOperator(dtype=float, shape=(*n, *n), matvec = matvec)
        return linop, rhs.flatten()

    def add_bcs(self, bcs: list[BoundaryCondition]):
        for bc in bcs:
            self._bcs.add(bc)
