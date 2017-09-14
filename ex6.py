# coding: utf-8


#$ header fd(double [:])
def fd(x):
    n = len(x)
    z = zeros(n, double)
    z = x+1
    y = 2 * z
    return y

xd = ones(6, double)
yd = zeros(6, double)
yd = 2.0 * fd(xd)
#t = fd(xd)
#yd = 2.0 * t
print(yd)
