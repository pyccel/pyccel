# pylint: disable=missing-function-docstring, missing-module-docstring

def incr_(x : int):
    def decr_(y : int):
        y = y-1
        return y
    x = x + 1
    return x

def helloworld():
    print('hello world')

def incr(x : int):
    x = x + 1
    return x

def decr(x : int) -> int:
    y = x - 1
    return y

def incr_array(x : 'int[:]'):
    x[:] = x + 1

y_=[1,2,3]

# def decr_array(x : int) -> int:
#     y_[1] = 6
#     z = y_
#     t = y_+x
#     return t

def decr_array(x : 'int[:]'):
    x[:] = x - 1

def f1(x : int, n : int = 2, m : int = 3) -> int:
    y = x - n*m
    return y

def f2(x : int, m : int = None):
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y

def my_print(a : 'int[:]'):
    print(a)


y = decr(2)
z = f1(1)

z1 = f2(1)
z2 = f2(1, m=0)

helloworld()

def pass_fun():
    pass

if __name__ == '__main__':
    print(y_)
    print(y)
    print(z)
    print(z1)
    print(z2)
    my_print([1,2,3])
