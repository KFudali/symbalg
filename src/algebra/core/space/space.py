from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .domain import TDomain

class Space(ABC, Generic[TDomain]):
    def __init__(self, domain: TDomain):
        self._domain = domain

    @property
    def domain(self) -> TDomain:
        return self._domain

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

TSpace = TypeVar("TSpace", bound=Space)
