from dataclasses import dataclass
from typing import Union
import enum
import numpy as np

from .boundary import BoundaryId


class BCType(enum.IntEnum):
    DIRICHLET = enum.auto()
    NEUMANN = enum.auto()


BCValue = Union[float, np.ndarray]


@dataclass(frozen=True)
class BoundaryCondition:
    bc_type: BCType
    value: BCValue
    boundary: BoundaryId
