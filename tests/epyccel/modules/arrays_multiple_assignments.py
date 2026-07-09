# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import stack_array

@stack_array("y")
def no_reallocation():
    import numpy as np

    x = np.zeros((2, 5), dtype=float)
    x = np.ones((2, 5), dtype=float)

    y = np.zeros((2, 2, 1), dtype=int)
    y = np.ones((2, 2, 1), dtype=int)

    return x.sum() + y.sum()

def creation_in_if_heap(c: "float"):
    import numpy as np

    if c > 0.5:
        x = np.ones(2, dtype=int)
    else:
        x = np.ones(7, dtype=int)
    return x.sum()

def creation_in_if_heap_shape(c: "float"):
    import numpy as np

    if c > 0.5:
        x = np.ones(3, dtype=int)
    else:
        x = np.ones(7, dtype=int)

    y = x[1:-1]
    return y.sum()

@stack_array("x")
def stack_array_if(b: bool):
    import numpy as np

    if b:
        x = np.array([1, 2, 3])
    else:
        x = np.array([4, 5, 6])
    return x[0]
