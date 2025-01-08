# pylint: disable=missing-function-docstring, missing-module-docstring
# this file is used inside imports.py
# make sure that you update the imports.py file if needed

def helloworld():
    print('hello world')

def incr(x : int):
    x = x + 1

def decr(x : int) -> int:
    y = x - 1
    return y

# TODO [YG, 30.01.2020] function behavior in Python not correct:
#      must change to x += 1
#
def incr_array(x : 'int [:]'):
    x = x + 1

def decr_array(x : 'int [:]') -> 'int [:]':
    y = x - 1
    return y

# TODO [YG, 30.01.2020] function behavior in Python not correct:
#      must change to x -= 1
#
def decr_array_inplace(x : 'int [:]'):
    x = x - 1

def f1(x : int, n : int = 2, m : int = None) -> int:
    y = x - n
    return y

def f2(x : int, m : int = None) -> int:
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y

y = decr(2)
z = f1(1)

z1 = f2(1)
z2 = f2(1, m=0)

# TODO add messages. for the moment there's a bug in Print
print(z1)
print(z2)
