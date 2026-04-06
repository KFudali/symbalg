from discr.core import Discretization, DiscreteBCFactory, DiscreteOperatorsFactory
from .time import TimeSeries

class Space():
    def __init__(self, discretization: Discretization):
        self._discretization = discretization
        self._time = TimeSeries()

    @property
    def time(self) -> TimeSeries:
        return self._time

    @property
    def discretization(self) -> Discretization:
        return self._discretization
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._discretization.shape

    @property
    def bcs(self) -> DiscreteBCFactory:
        return self._discretization.bcs
    
    @property
    def operators(self) -> DiscreteOperatorsFactory:
        return self._discretization.operators
