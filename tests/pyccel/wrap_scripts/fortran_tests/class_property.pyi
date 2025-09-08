#$ header metavar libraries="class_property"
#$ header metavar libdirs="."

class Counter:
    @low_level('create')
    def __init__(self, start: int) -> None: ...

    @low_level('free')
    def __del__(self) -> None: ...

    @low_level('increment')
    def increment(self) -> None: ...

    @low_level('get_value')
    @property
    def value(self) -> int: ...

