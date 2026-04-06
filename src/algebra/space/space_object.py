from abc import ABC, abstractmethod
from typing import Generic
from .space import Space, TDomain


class SpaceObject(ABC, Generic[TDomain]):
    def __init__(self, space: Space[TDomain]):
        self._space = space

    @property
    def space(self) -> Space[TDomain]:
        return self._space
