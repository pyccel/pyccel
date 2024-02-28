# pylint: disable=missing-function-docstring, missing-module-docstring
#==============================================================================

from pyccel.decorators import kernel

# This kernel function increments the value of a in-place
@kernel
def increment_value_inplace(a : int):
    a += 1

# ...
@kernel
def make_it_ten(a: int):
    a = 10
def main():
    a = 1 
    make_it_ten[1, 1](a)
if __name__ == "__main__":
    main()
