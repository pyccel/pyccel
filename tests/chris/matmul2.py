from numpy import array, zeros, matmul, sum

A = array([[1.,2.,3.],[4.,5.,6.],[7.,7.,9.]])
x = array([1.,2.,3.])
v = zeros(3)
v = matmul(A,x)
print(v)


