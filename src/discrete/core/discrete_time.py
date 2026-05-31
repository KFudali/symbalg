from abc import ABC, abstractmethod
from tools.advanceable import AdvanceableSeries


class DiscreteTimeView(ABC):
    @abstractmethod
    def dt(self) -> float:
        pass

    @property
    @abstractmethod
    def current(self) -> float:
        pass


class DiscreteTime(DiscreteTimeView):
    def __init__(self, dt: float = 0.01):
        self._current = 0.0
        self._discrete_steps = [0.0]
        self._dt = dt
        self._dts = list[float]()
        self._advanceables = AdvanceableSeries()

    @property
    def current(self) -> float:
        return self._current

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
        self._current = 0.0
        self._discrete_steps = [0.0]
        self._dts.clear()

    @property
    def discrete_steps(self) -> list[float]:
        return self._discrete_steps

    @property
    def advanceables(self) -> AdvanceableSeries:
        return self._advanceables
