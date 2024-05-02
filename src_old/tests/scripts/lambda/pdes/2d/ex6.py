# coding: utf-8

# ... import symbolic tools
glt_function     = load('pyccel.symbolic.gelato', 'glt_function', True, 3)
weak_formulation = load('pyccel.symbolic.gelato', 'weak_formulation', True, 2)
dx               = load('pyccel.symbolic.gelato', 'dx', False, 1)
dy               = load('pyccel.symbolic.gelato', 'dy', False, 1)
# ...

# ... weak formulation
a = lambda x,y,u,v: (1+x**2+y**2) * dx(u) * dx(v) + dy(u) * dy(v)

wf        = weak_formulation(a, 2)
weak_form = lambdify(wf)
# ...

# ... computing the glt symbol
ga = glt_function(a, [4, 4], [2, 2])
g  = lambdify(ga)
# ...

# ... glt symbol evaluation at
#     (x,y) = (0.2,0.2) and (t1,t2) = (1.1,3.1)
y = g (0.2, 0.2, 1.1, 3.1)
# ...

print(' a          := ', a)
print(' wf         := ', wf)
print(' glt symbol := ', ga)
print('')
print(' symbol (0.2, 0.2, 1.1, 3.1) = ', y)
