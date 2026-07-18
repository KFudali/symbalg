from abc import ABC, abstractmethod
from typing import TypeVar

from algebra.space import Space
from .boundary import Boundary, BoundaryId
from .boundary_tool import BoundaryTool


class Domain(ABC):
    @property
    @abstractmethod
    def space(self) -> Space:
        pass

    @property
    @abstractmethod
    def boundaries(self) -> dict[BoundaryId, Boundary]:
        pass

    @abstractmethod
    def boundary(self, boundary_id: BoundaryId) -> Boundary:
        pass

    @property
    @abstractmethod
    def boundary_tool(self) -> BoundaryTool:
        pass


TDomain = TypeVar("TDomain", bound=Domain)
