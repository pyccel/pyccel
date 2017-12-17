# coding: utf-8

#$ header fd(double [:]) results(double [:])
def fd(x):
    z = zeros_like(x)
    z = x+1
    y = 2 * z
    return y

#$ header gd(double [:]) results(double [:])
def gd(x):
    y = 2 * x + 1
    return y

#$ header func(double) results(double)
def func(x):
    z = x+1
    y = 3 * z
    return y

n = 64
m = 5
a = ones(n, double)
b = ones(64, int)

f0 = 1.0
f1 = f0 + 2.0 * a[2]
f2 = a[2] + 2.0 * f1
f3 = f0 + 2.0 * func(a[2])
f4 = func(a[2]) + 2.0 * f0

a1 = zeros_like(a)
a2 = 2.0 * a + 1.0

c0 = zeros((m,n), double)
c1 = c0[0,1:3]
c2 = 2.0 * c0[0,1:3] + 1.0

d0 = zeros((m,n,n), double)
d1 = d0[0,1:3,0:4]
d2 = 2.0 * d0[0,1:3,0:4] + 1.0

r1 = dot(a, a)
r2 = 2.0 + 3.0 * dot(a, a)

i1 = dot(b, b)
i2 = 2 + 3 * dot(b, b)

#not working
#xd = ones(6, double)
#yd = zeros(6, double)
#yd = 2.0 * fd(xd) + 1.0
#yd = 3.0 * fd(2.0 * fd(xd) + 1.0) + 1.0

#not working
#x = ones(6, double)
#y = zeros(6, double)
#y = 2.0 * fd(x) + 1.0
#y = 3.0 * fd(2.0 * fd(x) + 1.0) + 1.0

#not working
#yd = gd(xd)
#y  = gd(x)

#not working
#r_x = list(range(4,8))
#xr = zeros(r_x, double)
