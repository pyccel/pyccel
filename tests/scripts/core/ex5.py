# coding: utf-8


#$ header function f(double, double) results(double)
def f(u,v):
    t = u - v
    return t

#$ header g(double, double)
def g(x,v):
    m = x - v
    t =  2.0 * m
    z =  2.0 * t
    return t, z

#$ header fi(int, int)
def fi(x, n=5):
    y = x+1+n
    return y

x1 = 1.0
y1 = 2.0

w    = 2 * f(x1,y1) + 1.0

z, t = g(x1,w)

i = 1
j = 3
k = fi(i, j)
k = fi(i)

print(z)
print(t)
