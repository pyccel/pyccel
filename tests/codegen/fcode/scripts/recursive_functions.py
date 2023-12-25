# pylint: disable=missing-function-docstring, missing-module-docstring
def fact(n : int) -> int:
    if n == 0:
       z = 1
       return z
    else:
       z = n*fact(n-1)
       return z

def qsort_kernel(a : "double[:]", lo : int, hi : int) :
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


