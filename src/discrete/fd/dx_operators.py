from tools.geometry import StructuredGridND
from discrete.core.dx import DxOperators
from algebra.operator import Operator
from algebra.space import Space

from .operators import dx


class FDDxOperators(DxOperators):
    def __init__(self, space: Space, grid: StructuredGridND):
        super().__init__()
        self._space = space
        self._grid = grid

    def laplace(self, order: int = 1) -> Operator:
        return dx.laplace(self._space, order, self._grid.spacing[0])

    def grad(self, order: int = 1) -> Operator:
        return dx.grad(self._space, order, self._grid.spacing[0])

    def div(self, order: int = 1) -> Operator:
        return dx.div(self._space, order, self._grid.spacing[0])
