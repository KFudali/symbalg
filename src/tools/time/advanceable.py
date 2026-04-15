from abc import ABC, abstractmethod

class Advanceable(ABC):
    @abstractmethod
    def advance(self):
        pass
