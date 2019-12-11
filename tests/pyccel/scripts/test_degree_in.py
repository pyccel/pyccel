from pyccel.decorators import types, external, stack_array, pure

@pure
@external
@types('int')
@stack_array('tmp')
def test_degree(degree):
    from numpy import empty

    tmp = empty(degree+1, dtype=float)
    for i in range(degree+1):
        tmp[i]=0.
    return 1

print(test_degree(3))
