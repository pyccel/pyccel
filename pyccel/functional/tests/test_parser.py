# coding: utf-8

# TODO

from pyccel.functional.parser import parse

ast = parse('lambda xs: [g(x) for x in xs]')
ast = parse('lambda xs: [g(h(x)) for x in xs]')
ast = parse('lambda xs,ys: [[g(h(x),y) for y in ys] for x in xs]')
ast = parse('lambda xs,ys: [[g(h(x),y) for y in ys for x in xs]]') # TODO do not accept
ast = parse('lambda xs,ys: [g(h(x),y) for y in ys for x in xs]')
ast = parse('lambda xs,ys: add([[g(h(x),y) for y in ys for x in xs]])')
