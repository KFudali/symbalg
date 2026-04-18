from algebra.core.space import TimeDim
from tools.advanceable import Advanceable, AdvanceableSeries

class DiscreteTime(TimeDim):
    def __init__(self, dt: float = 0.01):
        TimeDim.__init__(self)
        self._discrete_steps = [0.0]
        self._dt = dt
        self._dts = list[float]()
        self._advanceables = AdvanceableSeries()

    def set_dt(self, dt: float):
        self._dt = dt

    def dt(self) -> float:
        return self._dt

    def advance(self):
        self._advanceables.advance()
        dt = self.dt()
        self._current += dt
        self._dts.append(dt)
        self._discrete_steps.append(self._current)

    def reset(self):
        self._discrete_steps = [0.0]
        self._dts.clear()

    @property
    def discrete_steps(self) -> list[float]:
        return self._discrete_steps

    @property
    def advanceables(self) -> AdvanceableSeries:
        return self._advanceables
