from .advanceable import Advanceable

class AdvanceableSeries():
    def __init__(self):
        self._advanceables = set[Advanceable]()

    def register(self, advanceable: Advanceable):
        self._advanceables.add(advanceable)

    def unregister(self, advanceable: Advanceable):
        self._advanceables.remove(advanceable)

    def advance(self):
        for adv in self._advanceables:
            adv.advance()
