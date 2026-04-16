from algebra.core.space.domain import BoundaryId
from algebra.systems.bcs import BoundaryCondition, BCType

def dirichlet(boundary_id: BoundaryId, value: float):
    return BoundaryCondition(boundary_id, BCType.DIRICHLET, value)

def neumann(boundary_id: BoundaryId, value: float):
    return BoundaryCondition(boundary_id, BCType.NEUMANN, value)
