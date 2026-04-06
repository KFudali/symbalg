from abc import ABC, abstractmethod
from typing import TypeVar

from .boundary import Boundary, BoundaryId

class Domain(ABC):
    @abstractmethod
    def boundary(self, boundary_id: BoundaryId) -> Boundary:
        pass

    @property
    @abstractmethod
    def boundaries(self) -> dict[BoundaryId, Boundary]:
        pass

TDomain = TypeVar("TDomain", bound=Domain)
