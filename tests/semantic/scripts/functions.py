# pylint: disable=missing-function-docstring, missing-module-docstring/
# this file is used inside imports.py
# make sure that you update the imports.py file if needed

def helloworld():
    print('hello world')

#$ header function incr(int)
def incr(x):
    x = x + 1

#$ header function decr(int) results(int)
def decr(x):
    y = x - 1
    return y

# TODO [YG, 30.01.2020] function behavior in Python not correct:
#      must change to x += 1
#
#$ header function incr_array(int [:])
def incr_array(x):
    x = x + 1

##$ header function decr_array(int [:]) results(int [:])
#def decr_array(x):
#    y = x - 1
#    return y

# TODO [YG, 30.01.2020] function behavior in Python not correct:
#      must change to x -= 1
#
#$ header function decr_array(int [:])
def decr_array(x):
    x = x - 1

#$ header function f1(int, int, int) results(int)
def f1(x, n=2, m=None):
    y = x - n
    return y

#$ header function f2(int, int) results(int)
def f2(x, m=None):
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
