from pyccel.decorators import kernel
# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================
@kernel
def increment_value_inplace(a):
    # This kernel function increments the value of a in-place
    a += 1