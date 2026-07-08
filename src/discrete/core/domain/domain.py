from abc import ABC, abstractmethod
from typing import TypeVar

from algebra.space import Space
from .boundary import Boundary, BoundaryId


class Domain(ABC):
    @property
    @abstractmethod
    def space(self) -> Space:
        pass

    @abstractmethod
    def boundary(self, boundary_id: BoundaryId) -> Boundary:
        pass

    @property
    @abstractmethod
    def boundaries(self) -> dict[BoundaryId, Boundary]:
        pass


TDomain = TypeVar("TDomain", bound=Domain)
