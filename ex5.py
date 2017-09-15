# coding: utf-8


#$ header fd(double)
def fd(x):
    y = x+1
    return y

#$ header fi(int)
def fi(x):
    y = x+1
    return y

xd = double()
xd = 1
yd = fd(xd)

xi = int()
xi = 1
yi = fi(xi)
