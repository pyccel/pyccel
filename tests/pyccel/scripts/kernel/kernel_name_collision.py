# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import kernel

@kernel
def do():
    pass

do[1,1]()
