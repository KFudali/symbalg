from typing import Callable, Any


class LazyAction:
    def __init__(self, action_callable: Callable[[Any], Any], input: Any | None = None):
        self._callable = action_callable
        self._input = input

    def perform(self) -> Any:
        if self._input:
            self._callable(self._input)
        else:
            self._callable()
