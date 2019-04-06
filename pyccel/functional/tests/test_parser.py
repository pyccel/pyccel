# coding: utf-8

# TODO use pytest

from pyccel.functional.parser import parse

L = parse('lambda x: 1')
L = parse('lambda xs: map(g, xs)')
L = parse('lambda xs: map(g, par(xs))')
L = parse('lambda xs,ys:  map(g, product(xs,ys))')
L = parse('lambda xs,ys: tmap(g, product(xs,ys))')
L = parse('lambda xs,ys: mul(map(g, product(xs,ys)))')
L = parse('lambda xs,ys: mul(map(g, zip(xs,ys)))')
L = parse('lambda xs,ys: add(map(g, par(product(xs,ys))))')
L = parse('lambda xs,ys: add(map(g, par(zip(xs,ys))))')

# g is a function of 2 arguments => use abstract function
L = parse('lambda a,xs: map(lambda x: g(x,a), xs)')
L = parse('lambda a,xs: map(lambda _: g(a,_), xs)')
