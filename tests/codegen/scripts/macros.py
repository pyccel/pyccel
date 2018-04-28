#$ header function f(int, int [:])
#$ header macro __g__(x) := f(x.shape, x)
def f(n, a):
    for i in range(0, n):
        a[i] = i

from numpy import zeros

a = zeros(4, 'int')
__g__(a)

print('hello world')
