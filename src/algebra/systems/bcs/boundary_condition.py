import enum
from typing import Generic
from dataclasses import dataclass
from .boundary_id import BoundaryId


class BCType(enum.IntEnum):
    DIRICHLET = enum.auto()
    NEUMANN = enum.auto()


@dataclass(frozen=True)
class BoundaryCondition:
    bc_type: BCType
    value: float
    id: BoundaryId
