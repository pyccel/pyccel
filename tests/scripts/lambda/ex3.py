# coding: utf-8

glt_function = load('pyccel.symbolic.gelato', 'glt_function')
dx           = load('pyccel.symbolic.gelato', 'dx')
dy           = load('pyccel.symbolic.gelato', 'dy')

bracket = lambda u,v: dy(u)*dx(v) - dx(u)*dy(v)
b       = lambda u,v: dx(u) * dx(v) + bracket(u,v)

# TODO to fix: bracket is not processed. we should use subs
gb = glt_function(b)
print(gb)
