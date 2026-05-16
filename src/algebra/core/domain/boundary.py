from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class BoundaryId:
    key: int

    def is_valid(self) -> bool:
        return self.key >= 0


@dataclass(frozen=True)
class Boundary:
    id: BoundaryId


TBoundary = TypeVar("TBoundary", bound=Boundary)
