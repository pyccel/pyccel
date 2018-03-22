# coding: utf-8

# ... import symbolic tools
weak_formulation = load('pyccel.symbolic.gelato', 'weak_formulation', True, 2)
glt_function     = load('pyccel.symbolic.gelato', 'glt_function',     True, 3)

Grad  = load('pyccel.symbolic.gelato', 'Grad',  False, 1)
Curl  = load('pyccel.symbolic.gelato', 'Curl',  False, 1)
Div   = load('pyccel.symbolic.gelato', 'Div',   False, 1)

Cross = load('pyccel.symbolic.gelato', 'Cross', False, 2)
Dot   = load('pyccel.symbolic.gelato', 'Dot',   False, 2)
# ...

# ... Laplace
a1  = lambda x,y,z,v,u: Dot(Grad(u), Grad(v))

ga1 = glt_function(a1, [4, 4, 4], [2, 2, 2])
wa1 = weak_formulation(a1, 3)

print(' a1            := ', a1)
print(' glt symbol a1 := ', ga1)
print('wa1            := ', wa1)
print('')
# ...

# ...
a2  = lambda x,y,z,v,u: Div(u) * Div(v) + 0.2 * Dot(u, v)

ga2 = glt_function(a2, [4, 4, 4], [2, 2, 2])
wa2 = weak_formulation(a2, 3)

print(' a2            := ', a2)
print(' glt symbol a2 := ', ga2)
print('wa2            := ', wa2)
print('')
# ...

# ...
a3  = lambda x,y,z,v,u: Dot(Curl(u), Curl(v)) + 0.2 * Dot(u, v)

ga3 = glt_function(a3, [4, 4, 4], [2, 2, 2])
wa3 = weak_formulation(a3, 3)

print(' a3            := ', a3)
print(' glt symbol a3 := ', ga3)
print('wa3            := ', wa3)
print('')
# ...
