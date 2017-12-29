# coding: utf-8

glt_function = load('pyccel.symbolic.gelato', 'glt_function', True)
dx           = load('pyccel.symbolic.gelato', 'dx', False)
dy           = load('pyccel.symbolic.gelato', 'dy', False)

bracket = lambda u,v: dy(u)*dx(v) - dx(u)*dy(v)
b       = lambda u,v: dx(u) * dx(v) + bracket(u,v)

gb = glt_function(b)

print('> b := ', b)
print('> symbol := ', gb)
