from enum import Enum, auto
from dataclasses import dataclass
from algebra.space.domain import BoundaryId

class BCType(Enum):
    DIRICHLET = auto()
    NEUMANN = auto()

@dataclass(frozen=True)
class BoundaryCondition():
    boundary: BoundaryId
    bctype: BCType
    value: float

