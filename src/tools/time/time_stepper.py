from .series import TimeSeries
from .advanceable import Advanceable

class TimeStepper():
    def __init__(self, time_series: TimeSeries):
        self._series = time_series
        self._advanceables = set[Advanceable]()

    @property
    def series(self) -> TimeSeries:
        return self._series

    def register(self, advanceable: Advanceable):
        self._advanceables.add(advanceable)

    def unregister(self, advanceable: Advanceable):
        self._advanceables.remove(advanceable)

    def advance(self):
        self._series.advance()
        for adv in self._advanceables:
            adv.advance()
