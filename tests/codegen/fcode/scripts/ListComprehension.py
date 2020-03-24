
# Creates a 1d array
x = [i*j for i in range(1000) for j in range(0,i,2) for k in range(0,3)]

# Creates a 1d array
y = [5.]*50

# Creates a 1d array
z = [i*j*k1 for i in range(200) for j in range(0,i,2) for k1 in y]

# Creates a 2d array
s = [(x1, y1, z1)  for x1 in range(1,30) for y1 in range(x1,30) for z1 in range(y1,1000)]

# Raises an error
t = [(a1, False, c1)  for a1 in range(1,30) for b1 in range(a1,30) for c1 in range(b1,1000)]
