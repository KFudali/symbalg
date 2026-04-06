from typing import Protocol

class Advanceable(Protocol):
    def advance(self, dt: float): pass


class TimeSeries():
    def __init__(self):
        self._dts = []
        self._next_dt = 0.01
        self._time = 0.0
        self._advenceables = set[Advanceable]()

    def advance(self, dt: float):
        for advanceable in self._advenceables:
            advanceable.advance(dt)
        self._dts.append(dt)
        self._time += dt

    @property
    def dts(self) -> list[float]: 
        return self._dts

    def dt(self) -> float:
        return self._next_dt

    def last_dt(self) -> float:
        return self._dts[-1]
    
    def time(self) -> float:
        return self._time

    @property
    def advanceables(self) -> set[Advanceable]:
        return self._advenceables

    def register_advanceable(self, advanceable: Advanceable):
        self._advenceables.add(advanceable)

    def loop(self, start: float, stop: float, *, dt: float):
        t = start
        self._next_dt = dt
        self._time = start
        while t < stop:
            yield t
            self.advance(dt)
            t += dt
