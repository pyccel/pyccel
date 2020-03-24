ai = (1,4,5)
a,b,c = 1,False,3.0
d = a + 3
e = ai
f,g,h = ai
i = e[2]
ai_0 = 44

from numpy import ones
x = ones((2,3,2))
for z in range(2):
    for y in range(3):
        for w in range(2):
            x[z,y,w] = w+y*2+z*6
idx_0 = 1
idx = (1,idx_0,0)

print(ai)
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
print(ai_0)
print(x[idx])

idx_2 = (0,1,2)
print(x[idx,idx_2,1])

from pyccel.decorators import types, pure

@pure
@types('int','int')
def add2(x, y):
    return x+y

args = (3,4)
print(add2(*args))
