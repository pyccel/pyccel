from pyccel.decorators import external, pure
from .folder2_funcs import sum_to_n

@external
@pure
def testing():
    return sum_to_n(4)
