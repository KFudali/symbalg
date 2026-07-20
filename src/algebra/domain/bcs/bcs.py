from dataclasses import dataclass
from typing import Union
import enum
import numpy as np

from algebra.domain.boundary import BoundaryId

BCValue = Union[float, np.ndarray]


class BCType(enum.IntEnum):
    DIRICHLET = enum.auto()
    NEUMANN = enum.auto()


@dataclass(frozen=True)
class BoundaryCondition:
    value: BCValue
    bc_type: BCType
    boundary: BoundaryId
