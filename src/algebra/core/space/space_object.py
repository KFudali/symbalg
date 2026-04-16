from typing import Generic
from .space import TSpace

class SpaceObject(Generic[TSpace]):
    def __init__(self, space: TSpace):
        self._space = space

    @property
    def space(self) -> TSpace:
        return self._space
