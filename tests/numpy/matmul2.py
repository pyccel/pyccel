
import numpy as np
from numpy import array, zeros, matmul, sum

A = np.ones([3, 3])
B = np.ones([3, 2])
C = np.ones([2, 3])

A[1, 1] = 2
A[1, 2] = 2
B[2, 1] = 2
C[1, 2] = 2

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
print("=== w ===")
print(w[0, :])
print(w[1, :])
print(w[2, :])
print("=== y ===")
print(y[0, :])
print(y[1, :])

v2 = zeros(2)
v2 = matmul(x, B)
print("=== v2 ===")
print(v2)
