# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure

@pure
def const_int_float():
    return 1, 3.4

if __name__ == '__main__':
    a,b = const_int_float()
    print(a)
    print(b)

@pure
def const_complex_bool_int():
    return 1+2j, False, 8

if __name__ == '__main__':
    c, d, e = const_complex_bool_int()
    print(c)
    print(d)
    print(e)

@pure
def expr_complex_int_bool(n : 'int'):
    return 0.5+n*1j, 2*n, n==3

if __name__ == '__main__':
    f, g, h = expr_complex_int_bool(a)
    print(f)
    print(g)
    print(h)

def f3(x  : 'float' =  1.5, y  : 'float' =  2.5):
    return x+y, x-y

if __name__ == '__main__':
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

if __name__ == '__main__':
    print_multiple()

    #TODO remove comment when passing tuple as arguments is done in C
    #print(min(f3()))
    #print(max(f3()))


def f4(x : 'float', y  : 'float' =  2.5):
    x = x + y
    return x+y, x-y

if __name__ == '__main__':
    for k in range(2):
        print(f4(i,j))

    if (j>i):
        print(f4(i,j))

    k = 1
    while (k<3):
        k=k+1
        print(f4(i,j))

    print(i,j)

def print_func(x : int, y : int):
    print(x,y)

def test_issue910():
    x = 0
    y = 1
    print_func(x,y)
    return x,y

if __name__ == '__main__':
    l,m = test_issue910()
