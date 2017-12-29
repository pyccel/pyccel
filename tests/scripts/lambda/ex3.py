# coding: utf-8

glt_function = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
dx           = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy           = load('pyccel.symbolic.gelato', 'dy', False, 1)

bracket = lambda u,v: dy(u)*dx(v) - dx(u)*dy(v)
b       = lambda u,v: dx(u) * dx(v) + bracket(u,v)

gb = glt_function(b, [4, 4], [2, 2])

print('> b := ', b)
print('> symbol := ', gb)
