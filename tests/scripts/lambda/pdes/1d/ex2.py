# coding: utf-8

# ... import symbolic tools
glt_function = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
dx           = load('pyccel.symbolic.gelato', 'dx', False, 1)
# ...

# ... weak formulation
a  = lambda x,u,v: dx(u) * dx(v) + u * v
# ...

# ... computing the glt symbol
#     first the symbolic expression is computed then we 'lambdify' it
#     calling 'lambdify' will create a FunctionDef and then the 'expression'
#     will be available in the AST
ga = glt_function(a, 4, 2)
print(ga)

g = lambdify(ga)
y = g(0.5, 0.3)
# ...

# ... a Lambda expression can be printed
print(' a          := ', a)
print(' glt symbol := ', ga)
print('')
print(' symbol (0.5, 0.3) = ', y)
# ...
