#$ header metavar includes="__pyccel__mod__"
#$ header metavar libraries="array_methods"
#$ header metavar libdirs="."
import numpy as np
from pyccel.decorators import low_level

class ArrayOps:
    @low_level('create')
    def __init__(self) -> None: ...

    @low_level('free')
    def __del__(self) -> None: ...

    @low_level('set_data')
    def set_data(self, arr: 'float[:]', n: np.int32) -> None: ...

    @low_level('sum')
    def sum(self) -> float: ...

    @low_level('scale')
    def __imul__(self, factor: float) -> None: ...

