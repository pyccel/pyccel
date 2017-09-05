# coding: utf-8

x = stencil(shape=10, step=2)

cs = eval('pyccel.calculus', 'compute_stencil_uniform', (2, 4, 0.0, 0.25))

for i in range(0, 10):
    for k in range(-2, 2+1):
        x[i, k] = cs[k+2]
print(x)
