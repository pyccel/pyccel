# coding: utf-8


#$ header fd(double [:])
def fd(x):
    n = len(x)
    z = zeros(n, double)
    z = x+1
    y = 2 * z
    return y

##$ header gd(double [:])
#def gd(x):
#    y = 2 * x + 1
#    return y

xd = ones(6, double)
yd = zeros(6, double)
yd = 2.0 * fd(xd) + 1.0
print(yd)
yd = 3.0 * fd(2.0 * fd(xd) + 1.0) + 1.0
print(yd)
#yd = gd(xd)
#print(yd)
