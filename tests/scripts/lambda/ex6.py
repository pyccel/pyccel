# coding: utf-8

# TODO: - this example is not working yet
#       - fix 'g = lambdify(ga)'
#       - in print(ga), we should not see 'Lambda'

glt_function = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
dx           = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy           = load('pyccel.symbolic.gelato', 'dy', False, 1)

kappa_ = lambda x,y: x*y

a  = lambda u,v: kappa_ * dx(u) * dx(v) + dy(u) * dy(v)

kappa = lambdify(kappa_)

ga = glt_function(a, [4, 4], [2, 2])
#print(ga)

g = lambdify(ga)
