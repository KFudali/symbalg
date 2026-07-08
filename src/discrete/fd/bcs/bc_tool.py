import numpy as np

from algebra.bcs import BoundaryTool, BoundaryCondition
from algebra.systems.systems import LinearSystem
from discrete.fd.domain import FDBoundary
from discrete.fd.domain.fd_domain import FDDomain
from discrete.fd.operators import FDOperator
from discrete.fd.operators.bcs import apply, post_solve


class FDBCTool(BoundaryTool[FDOperator]):
    def __init__(self, domain: FDDomain):
        self._domain = domain

    def apply(
        self,
        bcs: list[BoundaryCondition],
        system: LinearSystem[FDOperator],
    ) -> LinearSystem[FDOperator]:
        return apply(self._domain, bcs, system)

    def post_solve(self, bcs: list[BoundaryCondition], field: np.ndarray) -> None:
        post_solve(self._domain, bcs, field)
