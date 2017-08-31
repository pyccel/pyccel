# coding: utf-8

#from numpy import zeros
#from numpy import linspace


#a = zeros(shape=64, dtype=float)
#b = zeros(shape=8, dtype=int)
#
#a[1] = 1.0
#a[2] = 1.0
#a[3] = 1.0
#
#c = a[1]
#
#d = c + 5.3 * a[1+1] + 4.0 - a[3]
#print(d)
#
##e = a # not working. e must be declared as an array
#
#x = zeros(shape=(2,8), dtype=float)
#x[1,1] = 1
#
#y = x[0,2]
#print(y)
#
#n = int()
#n = 2
#m = int()
#m = 3
#x = zeros(shape=(n,m,2), dtype=float)
#
#for i in range(0, n):
#    for j in range(0, m):
#        x[i,j,0] = i-j
#        x[i,j,1] = i+j
#
#print(x)


n = int()
n = 2
m = int()
m = 3
x = zeros(shape=(n,m), dtype=float)
y = zeros(shape=(n),   dtype=float)

y[:2] = x[:2,0] + 1
print(y)

