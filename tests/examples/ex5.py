# coding: utf-8


#$ header f(double, double)
def f(u,v):
    t = u - v
    return t

#$ header g(double, double)
def g(x,v):
    m = x - v
    t =  2.0 * m
    z =  2.0 * t
    return t, z

x1 = 1.0
y1 = 2.0

w    = 2 * f(x1,y1) + 1.0
z, t = g(x1,w)

print(z)
print(t)
