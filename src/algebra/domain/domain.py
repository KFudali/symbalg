from abc import ABC, abstractmethod
from typing import TypeVar

from algebra.space import Space
from .boundary import Boundary, BoundaryId
from .bcs import BoundaryTool


class Domain(ABC):
    @property
    @abstractmethod
    def space(self) -> Space:
        pass

    @property
    @abstractmethod
    def boundaries(self) -> dict[BoundaryId, Boundary]:
        pass

    @property
    @abstractmethod
    def bounary_tool(self) -> BoundaryTool:
        pass


TDomain = TypeVar("TDomain", bound=Domain)
