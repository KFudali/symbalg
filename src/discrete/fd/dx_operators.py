from tools.geometry import StructuredGridND
from discrete.core.dx_operators import DxOperators
from algebra.operator import Operator
from algebra.space import Space

from .operators import dx


class FDDxOperators(DxOperators):
    def __init__(self, space: Space, grid: StructuredGridND):
        super().__init__()
        self._space = space
        self._grid = grid

    def _eye(self) -> Operator:
        return dx.eye(self._space)

    def _laplace(self, order: int) -> Operator:
        return dx.laplace(self._space, order, self._grid.spacing[0])

    def _grad(self, order: int) -> Operator:
        return dx.grad(self._space, order, self._grid.spacing[0])

    def _div(self, order: int) -> Operator:
        return dx.div(self._space, order, self._grid.spacing[0])
