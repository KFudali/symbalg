from algebra.space import Space, SpaceObject, TDomain
from algebra.operator.operator import Operator


class SpaceOperator(SpaceObject[TDomain], Operator):
    def __init__(self, space: Space, input_components: int, output_components: int):
        SpaceObject.__init__(self, space)
        self._input_components = input_components
        self._output_components = output_components
        Operator.__init__(
            self,
            (input_components, *space.shape),
            (output_components, *space.shape),
        )

    @property
    def input_components(self) -> int:
        return self._input_components

    @property
    def output_components(self) -> int:
        return self._output_components
