import numpy as np
from discrete.core import Discretization
from algebra.expression import CallableExpression
from algebra.expression.symbolic import SymbolicExpression
from algebra.space import Shape


class TimeSeries:
    def __init__(self, discretization: Discretization):
        self._discrete_time = discretization.time
        self._space = discretization.space

    def run(self, duration: float, init_dt: float = 0.01):
        to_go = duration
        self._discrete_time.set_dt(init_dt)
        while to_go >= 0:
            yield self._discrete_time.current
            to_go -= self._discrete_time.dt()
            self._discrete_time.advance()

    def adapt_dt(self, dt: float):
        self._discrete_time.set_dt(dt)

    def reset(self):
        self._discrete_time.reset()

    def dt(self) -> SymbolicExpression:
        return SymbolicExpression.wrap(
            CallableExpression(
                Shape.scalar(self._space),
                lambda: np.array(self._discrete_time.dt()),
            )
        )
