# pylint: disable=missing-function-docstring, missing-module-docstring

type MyType = float

def set_i(x : 'MyType[:]', i : 'int', val : MyType):
    x[i] = val

def swap(x : 'MyType[:]', i : 'int', j : 'int'):
    temp = x[i]
    x[i] = x[j]
    x[j] = temp

def inplace_max(x : 'MyType[:]'):
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

def f(x : 'MyType[:]', y : 'MyType[:]'):
    n = x.shape[0]
    for i in range(n):
        set_i(x, i, float(i))
    for i in range(n//3):
        set_i(x[::3], i, -1.0)
    b = inplace_max(y[:])
    return b
