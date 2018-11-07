from numpy import ones, cross
b = ones((2,3))
for i in range(2):
    a[i,:] = (1,2,3)
    b[i,:] = (5,0,4)

c = cross(a,b)
print(c[0,:],c[1,:])
