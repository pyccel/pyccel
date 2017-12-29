# coding: utf-8

glt_function = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
dx           = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy           = load('pyccel.symbolic.gelato', 'dy', False, 1)

a  = lambda u,v: dx(u) * dx(v) + dy(u) * dy(v)
ga = glt_function(a, [4, 4], [2, 2])

g = lambdify(ga)
y = g(0.1, 0.3)

print('> symbol := ', ga)
print('> symbol(0.1, 0.3) = ', y)
