from numpy import zeros

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


a = zeros(4, 'int')
__f(a)
f1(5)
#f2(a)
v = __g(a)
v = g1(a)

print('hello world')
