# coding: utf-8

# This example is the python implementation of linear_in_loop.1.f from OpenMP 4.5 examples

# TODO: the original example from OpenMP is not working

n = 100
n = int()

n2 = n / 2
n2 = int()

a = zeros(n,  double)
b = zeros(n2, double)

for i in range(0, n):
    a[i] = i+1

j = 0
#$ omp parallel
#$ omp do linear(j:1)
for i in range(0, n, 2):
    j = j + 1
    b[j] = a[i] * 2.0
#$ omp end parallel

one = int()
one = 1
print((j, b(one), b(j)))
# print out: 50 2.0 198.0
