# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('double[:]','int','int')
def qsort_kernel ( a , lo , hi ) :
    i = lo
    j = hi
    while i < hi :
        pivot = a[ ( lo + hi ) // 2 ]
        while i <= j :
            while a[i] < pivot :
                i += 1
            while a[j] > pivot :
                j -= 1
            if i <= j :
                tmp = a [i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if lo < j :
            qsort_kernel( a , lo , j )
        lo = i
        j = hi

from numpy import array

x = array([0.66, 0.96, 0.06, 0.89, 0.57, 0.56, 0.08, 0.33, 0.04, 0.59])
qsort_kernel(x,0,9)

for xi in x:
    print(xi)
