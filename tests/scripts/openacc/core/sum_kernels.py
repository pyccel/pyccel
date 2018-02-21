# coding: utf-8

nx = 1024

vecA = zeros(nx, double)
vecB = zeros(nx, double)
vecC = zeros(nx, double)

# Initialization of vectors
for i in range(0, nx):
    vecA[i] = 1.0/(nx-i+1)
    vecB[i] = vecA[i]**2

#$ acc kernels
for i in range(0, nx):
    vecC[i] = vecA[i] * vecB[i]
#$ acc end kernels

# Compute the check value
c_sum = 0.0
for i in range(0, nx):
    c_sum += vecC[i]

print(' Reduction sum: ', c_sum)
