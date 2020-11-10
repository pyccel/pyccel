# pylint: disable=missing-function-docstring, missing-module-docstring/
x = 0
for i in range(0,4):
    x = 2 *i

x = 0
for i in range(0,10):
    x = x + 1
    y = 2*x
    x = x + 1
    for j in range(0,4):
        k = 2*y

n = 2
m = 3

from numpy import zeros
from numpy import sum as np_sum
z = zeros((n,m,2), 'double')

for i in range(0, n):
    for j in range(0, m):
        z[i,j,0] = i-j
        z[i,j,1] = i+j

print(np_sum(z))
t = zeros(n, 'double')
t[:2] = z[:2,0,0] + 1

print(np_sum(t))


x1 = [1, 2, 3]

for i in x1:
    print(i)

y1 = [4, 5, 6]

for i1,j1 in zip(x1, y1):
    print(i1,j1)

for i1,j1 in enumerate(x1):
    print(i1,j1)

from itertools import product
for i2,j2 in product(x1, y1):
    print(i2,j2)

#$ header function f(int)
def f(z):
    x= 5+z
    return x


mm = [1,2,3]
for ii in map(f,mm):
    print(ii)



