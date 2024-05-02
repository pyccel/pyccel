# coding: utf-8

x = stencil(shape=(100,100), step=(2,2))

c1 = eval('pyccel.calculus', 'compute_stencil_uniform', (2, 4, 0.0, 0.25))
c2 = eval('pyccel.calculus', 'compute_stencil_uniform', (2, 4, 0.0, 0.25))

for i1 in range(0, 100):
    for i2 in range(0, 100):
        for k1 in range(-2, 2+1):
            for k2 in range(-2, 2+1):
                x[i1, i2, k1, k2] = c1[k1+2] * c2[k2+2]
#print(x)
