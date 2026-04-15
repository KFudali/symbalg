from abc import ABC, abstractmethod

class TimeDim(ABC):
    @abstractmethod
    def dt(self) -> float:
        pass

    @abstractmethod
    def now(self) -> float:
        pass
