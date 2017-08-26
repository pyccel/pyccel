# coding: utf-8

def g(u,v):
    m = u - v;
    w =  2.0 * m;
    r =  2.0 * w;
    return r, m

x = 1.0
y = 2.0

z, t = g(x,y)


#def g(u,v):
#    t = u - v;
#    return t
#
#x = 1.0
#y = 2.0
#
#z = 2 * g(x,y) + 1.0
