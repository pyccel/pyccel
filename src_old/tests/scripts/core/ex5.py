# coding: utf-8


def f_pass():
    pass

#$ header fi(int, int) results(int)
def fi(x, n=5):
    y = x+1+n
    return y

#$ header function fd(double, double) results(double)
def fd(u,v):
    t = u - v
    return t

#$ header g(double, double) results(double, double)
def g(x,v):
    m = x - v
    t =  2.0 * m
    z =  2.0 * t
    return t, z


# ...
f_pass()
# ...

# ...
i = 1
j = 3

k = fi(i, j)
k = fi(i)
# ...

# ...
xd = 1.0
yd = 2.0

w = fd(xd,yd)
# ...

z = 0.0
t = 0.0

# ...
[z, t] = g(xd,w)

print(z)
print(t)

xr = random()
print(xr)
