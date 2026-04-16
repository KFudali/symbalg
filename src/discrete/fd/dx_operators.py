from discrete.core.dx import DxOperators
from discrete.fd.domain import FDDomain
from algebra.core.space import Space
from . import dx

class FDDxOperators(DxOperators):
    def __init__(self, space: Space[FDDomain]):
        super().__init__()
        self._space = space

    def laplace(self, components: int) -> dx.LapStencilOperator:
        return dx.laplace(self._space, components)

    def grad(self, components: int) -> dx.GradStencilOperator:
        return dx.grad(self._space, components)

    def div(self, components: int) -> dx.DivStencilOperator:
        return dx.div(self._space, components)
