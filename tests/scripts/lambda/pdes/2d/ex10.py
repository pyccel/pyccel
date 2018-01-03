# coding: utf-8

# ... import symbolic tools
Grad  = load('pyccel.symbolic.gelato', 'Grad',  False, 1)
Curl  = load('pyccel.symbolic.gelato', 'Curl',  False, 1)
Div   = load('pyccel.symbolic.gelato', 'Div',   False, 1)
Rot   = load('pyccel.symbolic.gelato', 'Rot',   False, 1)

Cross = load('pyccel.symbolic.gelato', 'Cross', False, 2)
Dot   = load('pyccel.symbolic.gelato', 'Dot',   False, 2)
# ...

# ... weak formulation
a1 = lambda x,y,u,v: Dot(Grad(u), Grad(v))
a2 = lambda x,y,u,v: Rot(u) * Rot(v) + 0.2 * Div(u) * Div(v)
# ...

# ... a Lambda expression can be printed
print(' a1 := ', a1)
print(' a2 := ', a2)
# ...
