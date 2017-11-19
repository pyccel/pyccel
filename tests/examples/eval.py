# coding: utf-8

n = 5
m = 3

x = eval('numpy', 'sqrt', (25))
print(("sqrt(25) = ", x))

y = eval('numpy', 'max', ((25,4)))
print(("max((25,4)) = ", y))

z = eval('numpy', 'linspace', (0.0,1.0,5))
print(("linspace(0.0, 1.0, 5) = ", z))

u,v = eval('ex5', 'g', (2,5))
print(("g(2, 3) = ", u, v))

cs = eval('pyccel.calculus', 'compute_stencil_uniform', (1, 4, 0.0, 0.25))
print(("stencil(2, 4, 0., 0.25) = ", cs))
