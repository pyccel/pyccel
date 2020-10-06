# pylint: disable=missing-function-docstring, missing-module-docstring/
def helloworld():
    print('hello world')

#$ header function incr(int|double)
def incr(x):
    x = x + 1

#$ header function decr(int|double)
def decr(x):
    y = x - 1
    return y

# TODO [YG, 30.01.2020] function behavior in Python not correct:
#      must change to x += 1
#
#$ header function incr_array(int [:]|double[:])
def incr_array(x):
    x = x + 1

y_=[1,2,3]

##$ header function decr_array([int]|[double])
#def decr_array(x):
#    y_[1] = 6
#    z = y_
#    t = y_+x
#    return t

# TODO [YG, 30.01.2020] function behavior in Python not correct:
#      must change to x -= 1
#
#$ header function decr_array(int [:]|double[:])
def decr_array(x):
    x = x - 1

#$ header function f1(int, int, int)
def f1(x, n=2, m=3):
    y = x - n*m
    return y

#$ header function f2(int, int)
def f2(x, m=None):
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y

y = decr(2)
z = f1(1)
t = 1.
print(z)
incr(z)
print(z)
print(t)
incr(t)
print(t)

z1 = f2(1)
z2 = f2(1, m=0)

# TODO add messages. for the moment there's a bug in Print
print(z1)
print(z2)
