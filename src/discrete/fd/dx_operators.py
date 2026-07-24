import numpy as np

from algebra.domain import SymbolicDomainOperator
from discrete.core import DxOperators

from .domain import FDDomain
from .operator import dx, FDDomainOperator
from . import stencil


class FDDxOperators(DxOperators):
    def __init__(self, domain: FDDomain):
        super().__init__()
        self._domain = domain

    def _make_symbolic(
        self, operator: stencil.StencilOperator
    ) -> SymbolicDomainOperator:
        domain_operator = FDDomainOperator(self._domain, operator)
        return SymbolicDomainOperator.wrap(domain_operator)

    def eye(self) -> SymbolicDomainOperator:
        return self._make_symbolic(dx.eye(self._domain.space))

    def laplace(self, order: int = 2) -> SymbolicDomainOperator:
        return self._make_symbolic(
            dx.laplace(self._domain.space, order, self._domain.grid.spacing[0])
        )

    def grad(self, order: int = 2) -> SymbolicDomainOperator:
        return self._make_symbolic(
            dx.grad(self._domain.space, order, self._domain.grid.spacing[0])
        )

    def div(self, order: int = 2) -> SymbolicDomainOperator:
        return self._make_symbolic(
            dx.div(self._domain.space, order, self._domain.grid.spacing[0])
        )

    def array(self, weights: np.ndarray) -> SymbolicDomainOperator:
        pass
