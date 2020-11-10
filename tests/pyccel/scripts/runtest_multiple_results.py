# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import pure, types

@pure
def const_int_float():
    return 1, 3.4

a,b = const_int_float()
print(a)
print(b)

@pure
def const_complex_bool_int():
    return 1+2j, False, 8

c, d, e = const_complex_bool_int()
print(c)
print(d)
print(e)

@pure
@types('int')
def expr_complex_int_bool(n):
    return 0.5+n*1j, 2*n, n==3

f, g, h = expr_complex_int_bool(a)
print(f)
print(g)
print(h)

@types('real','real')
def f3(x = 1.5, y = 2.5):
    return x+y, x-y

i,j = f3(19.2,6.7)
print(i,j)
i,j = f3(4.5)
print(i,j)
i,j = f3(y = 8.2)
print(i,j)
i,j = f3()
print(i, j)



print(f3())

def print_multiple():
    print(f3())

print_multiple()

#TODO remove comment when passing tuple as arguments is done in C
#print(min(f3()))
#print(max(f3()))


@types('real','real')
def f4(x, y = 2.5):
    x = x + y
    return x+y, x-y

for k in range(2):
    print(f4(i,j))

if (j>i):
    print(f4(i,j))

k = 1
while (k<3):
    k=k+1
    print(f4(i,j))

print(i,j)
