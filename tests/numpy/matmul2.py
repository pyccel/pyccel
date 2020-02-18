
import numpy as np
from numpy import array, zeros, matmul, sum

A = array([[1., 2., 3.], [4., 5., 6.], [7., 7., 9.]])
B = array([[1., 2.], [4., 5.], [7., 7.]])
C = array([[1., 2., 3.], [4., 5., 6.]])

# There is still a bug with rectangular matrix assignment,
# see https://github.com/pyccel/pyccel/issues/236
print("=== A ===")
print(A[0, :])
print(A[1, :])
print(A[2, :])
print("=== B ===")
print(B[0, :])
print(B[1, :])
print(B[2, :])
print("=== C ===")
print(C[0, :])
print(C[1, :])
x = array([1., 2., 3.])
v = zeros(3)
w = zeros([3, 3])
y = zeros([2, 2])
v = matmul(A, x)
w = matmul(A, A)
y = matmul(C, B)
#v[1] = sum(x)
print("=== v ===")
print(v)
print("")
print(w[0, :])
print(w[1, :])
print(w[2, :])
print("")
print(y[0, :])
print(y[1, :])
