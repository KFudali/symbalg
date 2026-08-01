import numpy as np
from algebra.field import Field
from algebra.space import Shape
from discrete.core import Discretization
from tools.buffer import DequeValueBuffer


class FieldFactory:
    def __init__(self, discrete: Discretization):
        self._discrete = discrete

    def _field(self, components: tuple[int, ...], init_value: float) -> Field:
        shape = Shape(self._discrete.space, components)
        value_buffer = DequeValueBuffer(shape.fieldshape())
        init = np.ones(shape=shape.fieldshape(), dtype=float) * init_value
        value_buffer.set(init)
        self._discrete.time.advanceables.register(value_buffer)
        return Field(shape, value_buffer)

    def scalar(self, init_value: float = 0.0) -> Field:
        return self._field((), init_value)

    def vector(self, init_value: float = 0.0) -> Field:
        return self._field((self._discrete.space.ndim,), init_value)

    def tensor(self, init_value: float = 0.0) -> Field:
        return self._field(
            (self._discrete.space.ndim, self._discrete.space.ndim), init_value
        )

    def custom(self, components: tuple[int, ...], init_value: float = 0.0) -> Field:
        return self._field(components, init_value)
