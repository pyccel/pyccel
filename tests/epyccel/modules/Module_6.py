# pylint: disable=missing-function-docstring, missing-module-docstring

def set_i(x : 'float[:]', i : 'int', val : 'float'):
    x[i] = val

def swap(x : 'float[:]', i : 'int', j : 'int'):
    temp = x[i]
    x[i] = x[j]
    x[j] = temp

def inplace_max(x : 'float[:]'):
    n = x.shape[0]
    # bubble sort
    for j in range(n):
        i = n-1-j
        while i < n-1:
            if x[i] > x[i+1]:
                swap(x, i, i+1)
            else:
                i+=1
    return x[-1]

def f(x : 'float[:]', y : 'float[:]'):
    n = x.shape[0]
    for i in range(n):
        set_i(x, i, float(i))
    for i in range(n//3):
        set_i(x[::3], i, -1.0)
    b = inplace_max(y[:])
    return b
