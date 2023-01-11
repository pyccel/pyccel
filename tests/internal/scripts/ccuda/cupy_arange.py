# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring/

from pyccel.decorators import kernel, types
import cupy as cp

if __name__ == '__main__':
    arr = cp.arange(32)
