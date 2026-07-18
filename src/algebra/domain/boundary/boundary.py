from dataclasses import dataclass
from typing import TypeVar

from .boundary_id import BoundaryId


@dataclass(frozen=True)
class Boundary:
    boundary: BoundaryId


TBoundary = TypeVar("TBoundary", bound=Boundary)
