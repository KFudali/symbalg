from typing import Protocol


class Advanceable(Protocol):
    def advance(self):
        pass

    def reset(self):
        pass
