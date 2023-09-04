# pylint: disable=missing-function-docstring, missing-module-docstring

def incr_(x : int):
    def decr_(y : int): # pylint: disable=unused-variable
        y = y-1
    x = x + 1

def helloworld():
    print('hello world')

def incr(x : int):
    x = x + 1

def decr(x : int):
    y = x - 1
    return y

# TODO [YG, 30.01.2020] function behavior in Python is not correct:
#      must change to x += 1
#
def incr_array(x : 'int[:]'):
    x = x + 1

y_=[1,2,3]

#def decr_array(x : '[int]'):
#    y_[1] = 6
#    z = y_
#    t = y_+x
#    return t

# TODO [YG, 30.01.2020] function behavior in Python is not correct:
#      must change to x -= 1
#
def decr_array(x : 'int[:]'):
    x = x - 1

def f1(x : int, n : int = 2, m : int = 3):
    y = x - n*m
    return y

def f2(x : int, m : int = None):
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y

y = decr(2)
z = f1(1)

z1 = f2(1)
z2 = f2(1, m=0)

helloworld()

# TODO add messages. for the moment there's a bug in Print
print(z1)
print(z2)
