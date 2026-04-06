from abc import ABC, abstractmethod
from .space import Space


class SpaceObject(ABC):
    def __init__(self, space: Space):
        self._space = space

    @property
    def space(self) -> Space:
        return self._space
