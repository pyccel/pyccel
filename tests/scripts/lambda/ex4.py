# coding: utf-8

glt_function = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
dx           = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy           = load('pyccel.symbolic.gelato', 'dy', False, 1)

laplace = lambda u,v: dx(u)*dx(v) + dy(u)*dy(v)
a       = lambda u,v: laplace(u,v) + 0.1 * dx(u) * v

ga = glt_function(a, [4, 4], [2, 2])
g = lambdify(ga)

# glt symbol is supposed to be 'complex' in this example
# TODO fix it. for the moment the symbol is always 'double'
y = g(0.1, 0.3)

print('> symbol := ', ga)
print('> symbol(0.1, 0.3) = ', y)
