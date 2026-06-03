import enum
from typing import Generic
from dataclasses import dataclass
from discrete.core.domain.boundary import TBoundary


class BCType(enum.IntEnum):
    DIRICHLET = enum.auto()
    NEUMANN = enum.auto()


@dataclass(frozen=True)
class BoundaryCondition(Generic[TBoundary]):
    bc_type: BCType
    value: float
    boundary: TBoundary
