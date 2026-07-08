import enum
from dataclasses import dataclass
from typing import Union
import numpy as np
from .boundary_id import BoundaryId


class BCType(enum.IntEnum):
    DIRICHLET = enum.auto()
    NEUMANN = enum.auto()


BCValue = Union[float, np.ndarray]


@dataclass(frozen=True)
class BoundaryCondition:
    bc_type: BCType
    value: BCValue
    id: BoundaryId
