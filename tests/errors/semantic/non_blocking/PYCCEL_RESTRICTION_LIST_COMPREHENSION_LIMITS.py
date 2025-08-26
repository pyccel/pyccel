# Pyccel cannot handle this list comprehension. This is because there are occasions where the upper bound is smaller than the lower bound for variable k
# pylint: disable=missing-function-docstring, missing-module-docstring
a = [i*j for i in range(1,3) for j in range(1,4) for k in range(i,j)]

n = 5

a = [i*j for i in range(1,n) for j in range(1,4) for k in range(i,j)]
