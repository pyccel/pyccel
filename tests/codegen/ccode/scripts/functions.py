# pylint: disable=missing-function-docstring, missing-module-docstring/
# pylint: disable=unused-variable

def helloworld():
    print('hello world')

#$ header function incr(int)
def incr(x):
    x = x + 1

#$ header function decr(int) results(int)
def decr(x):
    y = x - 1
    return y

#$ header function f1(int, int, int) results(int)
def f1(x, n=2, m=3):
    y = x - n*m
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

helloworld()

# TODO add messages. for the moment there's a bug in Print
print(z1)
print(z2)

def pass_fun():
    pass

