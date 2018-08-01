from numpy import array, zeros

A = array([[1.,2.,3.],[4.,5.,6.],[7.,7.,9.]])
x = array([1.,2.,3.])
v = zeros(3)

for k in range(3):
    for l in range(3):
        v[k] = v[k]+A[k,l]*x[l]

print(v)
