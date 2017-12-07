# coding: utf-8


#$ header fd(*double [:]) results(*double [:])
def fd(x):
    z = zeros_like(x)
    z = x+1
    r = 2 * z
    return r

#$ header gd(*double [:]) results(*double [:])
def gd(x):
    s = 2 * x + 1
    return s

xd = ones(6, double)
yd = zeros(6, double)
#yd = 2.0 * fd(xd) + 1.0
#print(yd)
#yd = 3.0 * fd(2.0 * fd(xd) + 1.0) + 1.0
#print(yd)
#
#yd = gd(xd)
#print(yd)
