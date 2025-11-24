#$ header metavar includes="__pyccel__mod__"
#$ header metavar libraries="class_overloaded_methods"
#$ header metavar libdirs="."
from typing import overload

class Adder:
    def __init__(self) -> None: ...
    def __del__(self) -> None: ...

    @overload
    def add(self, x: int, y: int) -> int: ...
    @overload
    def add(self, x: float, y: float) -> float: ...

