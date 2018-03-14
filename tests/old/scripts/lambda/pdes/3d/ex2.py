# coding: utf-8

# ... import symbolic tools
glt_function = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
dx           = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy           = load('pyccel.symbolic.gelato', 'dy', False, 1)
dz           = load('pyccel.symbolic.gelato', 'dz', False, 1)
# ...

# ... weak formulation
a  = lambda x,y,z,u,v: dx(u) * dx(v) + dy(u) * dy(v) + dz(u) * dz(v)
# ...

# ... computing the glt symbol
#     first the symbolic expression is computed then we 'lambdify' it
#     calling 'lambdify' will create a FunctionDef and then the 'expression'
#     will be available in the AST
ga = glt_function(a, [4, 4, 4], [2, 2, 2])

g = lambdify(ga)
y = g(0.5, 0.5, 0.1, 0.5, 0.1, 0.3)
# ...

# ... a Lambda expression can be printed
print(' a          := ', a)
print(' glt symbol := ', ga)
print('')
print(' symbol (0.5, 0.5, 0.5, 0.1, 0.1, 0.3) = ', y)
# ...
