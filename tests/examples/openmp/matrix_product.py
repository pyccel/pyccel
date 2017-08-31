# coding: utf-8

#from numpy import zeros

n = int()
m = int()
p = int()
n = 2
m = 4
p = 2

a = zeros(shape=(n,m), dtype=float)
b = zeros(shape=(m,p), dtype=float)
c = zeros(shape=(n,p), dtype=float)

x = 0
y = 0

##@ omp parallel private (x, y)
#for i in range(0, n):
#    for j in range(0, m):
#        a[i,j] = i-j
##@ omp end parallel


#for i in range(0, n):
#    for j in range(0, m):
#        a[i,j] = i-j
#
#for i in range(0, m):
#    for j in range(0, p):
#        b[i,j] = i+j

##@ omp parallel private (x, y)
#@ omp do schedule(runtime)
for i in range(0, n):
    for j in range(0, p):
        for k in range(0, p):
            c[i,j] = c[i,j] + a[i,k]*b[k,j]
##@ omp end do
##@ omp end parallel

print(c)
