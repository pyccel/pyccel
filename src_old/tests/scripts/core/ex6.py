# coding: utf-8


a = zeros(64, double)
b = zeros(8, int)

a[1] = 1.0
a[2] = 1.0
a[3] = 1.0

c = a[1]

d = c + 5.3 * a[1+1] + 4.0 - a[3]
print(d)

e = zeros((2,8), double)
e[1,1] = 1

# not working
#f = e[0,2]
#print(f)

n = 2
m = 3
x = zeros((n,m,2), double)

for i in range(0, n):
    for j in range(0, m):
        x[i,j,0] = i-j
        x[i,j,1] = i+j
print(x)

y = zeros(n, double)
y[:2] = x[:2,0,0] + 1
print(y)

