#$ header metavar includes="."
#$ header metavar libraries="class_overloaded_methods"
#$ header metavar libdirs="."
from typing import overload

class Adder:
    @low_level('Adder__create')
    def __init__(self) -> None: ...

    @low_level('Adder__free')
    def __del__(self) -> None: ...

    @low_level('Adder__add_0000')
    @overload
    def add(self, x: int, y: int) -> int: ...

    @low_level('Adder__add_0001')
    @overload
    def add(self, x: float, y: float) -> float: ...

