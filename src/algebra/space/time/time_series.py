from abc import ABC, abstractmethod

class TimeSeries(ABC):
    @abstractmethod
    def next_dt(self) -> float:
        pass

    @property
    def range(self) -> tuple[float, float]:
        pass
