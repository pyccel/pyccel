from pyccel.decorators import kernel
@kernel
def increment_value_inplace(a):
    # This kernel function increments the value of a in-place
    a += 1
