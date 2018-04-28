from numpy import zeros

# .....................................
#           1d case
# .....................................
#$ header function f(int, int [:])
#$ header macro __f(x) := f(x.shape, x)
def f(n, a):
    for i in range(0, n):
        a[i] = i

#$ header function f1(int)
def f1(n):
    x = zeros(n, 'int')
    __f(x)

# TODO not working yet, x is intent(inout)
#      but it is not infered as inout
##$ header function f2(int [:])
#def f2(x):
#    __f(x)

#$ header function g(int, int [:]) results(int)
#$ header macro __g(x) := g(x.shape, x)
def g(n, a):
    r = 0
    for i in range(0, n):
        r += a[i]
    return r

#$ header function g1(int [:]) results(int)
def g1(x):
    v = __g(x)
    return v
# .....................................


# .....................................
#           2d case
# .....................................
#$ header function f2d(int [:], int [:,:])
#$ header macro __f2d(x) := f2d(x.shape, x)
def f2d(nm, a):
    for i in range(0, nm[0]):
        for j in range(0, nm[1]):
            a[i,j] = i*j

# TODO not working yet
#      gfortran error: ‘nm’ must be ALLOCATABLE
##$ header function f2d1(int, int)
#def f2d1(n,m):
#    x = zeros((n, m), 'int')
#    __f2d(x)

#$ header function k2d(int, int, int [:,:])
#$ header macro __k2d(x) := k2d(x.shape[0], x.shape[1], x)
def k2d(n, m, a):
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i*j

#$ header function k2d1(int, int)
def k2d1(n, m):
    x = zeros((n, m), 'int')
    __k2d(x)

# .....................................


# ... 1d array
a = zeros(4, 'int')
__f(a)
f1(5)
# TODO not working yet
#f2(a)
v = __g(a)
v = g1(a)
# ...

# ... 2d array
b = zeros((4,3), 'int')
# TODO not working yet
#      gfortran error: ‘nm’ must be ALLOCATABLE
#__f2d(b)

__k2d(b)
k2d1(3,5)
# ...

print('hello world')
