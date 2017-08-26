# coding: utf-8

def f(u,v):
    t = u - v
    return t

def g(x,v):
    m = x - v
    t =  2.0 * m
    z =  2.0 * t
    return t, z

x = 1.0
y = 2.0

w    = 2 * f(x,y) + 1.0
z, t = g(x,w)
