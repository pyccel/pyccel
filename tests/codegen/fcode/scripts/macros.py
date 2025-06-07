# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import zeros

# .....................................
#           1d case
# .....................................
#$ header macro __f(x) := f(x.shape, x)
def f(n : int, a : 'int[:]'):
    for i in range(0, n):
        a[i] = i

def f1(n : int):
    x = zeros(n, 'int')
    __f(x)

def f2(x : 'int[:]'):
    __f(x) # pylint: disable=undefined-variable
# .....................................

# .....................................
#           2d case
# .....................................
#$ header macro __f2d(x) := f2d(x.shape, x)
def f2d(nm : 'tuple[int, int]', a : 'int[:,:]'):
    for i in range(0, nm[0]):
        for j in range(0, nm[1]):
            a[i,j] = i*j

def f2d1(n : int, m : int):
    x = zeros((n, m), 'int')
    __f2d(x)

#$ header macro __k2d(x) := k2d(x.shape[0], x.shape[1], x)
def k2d(n : int, m : int, a : 'int[:,:]'):
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i*j

def k2d1(n : int, m : int):
    x = zeros((n, m), 'int')
    __k2d(x)

# .....................................

# .....................................
#       macros with results
# .....................................
#$ header macro (y), __h(x) := h(x.shape, x, y)

def h(n : int, a : 'int[:]', b : 'int[:]'):
    b[:] = a[:]
# .....................................


# ... 1d array
a = zeros(4, 'int')
__f(a)
f1(5)
# TODO not working yet
#f2(a)
# ...

# ... 2d array
b = zeros((4,3), 'int')
# TODO not working yet
#      gfortran error: ‘nm’ must be ALLOCATABLE
#__f2d(b)

__k2d(b)
k2d1(3,5)
# ...

# ... macros with results
a1 = zeros(4, 'int')
b1 = zeros(4, 'int')
b1 = __f(a1)
# ...

print('hello world')
