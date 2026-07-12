from abc import ABC, abstractmethod
from typing import TypeVar

from .bcs.boundary import Boundary
from .bcs.boundary_id import BoundaryId


class Domain(ABC):
    @abstractmethod
    def boundary(self, boundary_id: BoundaryId) -> Boundary:
        pass

    @property
    @abstractmethod
    def boundaries(self) -> dict[BoundaryId, Boundary]:
        pass


TDomain = TypeVar("TDomain", bound=Domain)
