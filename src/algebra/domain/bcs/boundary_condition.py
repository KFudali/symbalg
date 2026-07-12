import enum
from dataclasses import dataclass
from typing import Union, Generic
import numpy as np
from .boundary import TBoundary


class BCType(enum.IntEnum):
    DIRICHLET = enum.auto()
    NEUMANN = enum.auto()


BCValue = Union[float, np.ndarray]


@dataclass(frozen=True)
class BoundaryCondition(Generic[TBoundary]):
    bc_type: BCType
    value: BCValue
    boundary: TBoundary
