
from numpy import array

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
