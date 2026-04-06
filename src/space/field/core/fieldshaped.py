from algebra.space import SpaceObject, Space

class FieldShaped(SpaceObject):
    def __init__(self, space: Space, components: int):
        super().__init__(space)
        self._space = space
        self._components = components

    @property
    def space(self) -> Space:
        return self._space

    @property
    def components(self) -> int:
        return self._components

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.components, *self.space.shape)
