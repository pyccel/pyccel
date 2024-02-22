# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

from pyccel.decorators import kernel
# This kernel function increments the value of a in-place
@kernel
def increment_value_inplace(a):
    a += 1

