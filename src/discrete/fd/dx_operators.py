import numpy as np
from discrete.fd.domain import FDDomain
from discrete.core.dx_operators import DxOperators
from algebra.operator import Operator

from .operators import dx


class FDDxOperators(DxOperators):
    def __init__(self, domain: FDDomain):
        super().__init__()
        self._domain = domain

    def _eye(self) -> Operator:
        return dx.eye(self._domain)

    def _laplace(self, order: int) -> Operator:
        return dx.laplace(self._domain, order)

    def _grad(self, order: int) -> Operator:
        return dx.grad(self._domain, order)

    def _div(self, order: int) -> Operator:
        return dx.div(self._domain, order)

    def _array(self, weights: np.ndarray) -> Operator:
        pass
