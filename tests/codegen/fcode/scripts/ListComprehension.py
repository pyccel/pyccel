x = [i*j for i in range(1000) for j in range(0,i,2) for k in range(0,3)]

y = [5.]*50

z = [i*j*k1 for i in range(200) for j in range(0,i,2) for k1 in y]

s = [(x1, y1, z1)  for x1 in range(1,30) for y1 in range(x1,30) for z1 in range(y1,1000)]
