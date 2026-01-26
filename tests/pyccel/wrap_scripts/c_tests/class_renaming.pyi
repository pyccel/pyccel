#$ header metavar includes="."
#$ header metavar libraries="class_renaming"
#$ header metavar libdirs="."

@low_level('class_renaming__Counter')
class Counter:
    @low_level('Counter__create')
    def __init__(self, start: int) -> None: ...

    @low_level('Counter__free')
    def __del__(self) -> None: ...

    @low_level('Counter__increment')
    def increment(self) -> None: ...

    @low_level('Counter__get_value')
    @property
    def value(self) -> int: ...

