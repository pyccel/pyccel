from pyccel.decorators import external, pure
from ..folder1.folder1_funcs import sum_to_n

@external
@pure
def test_func():
    return sum_to_n(4)
