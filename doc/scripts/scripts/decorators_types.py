@types(int)
def incr_(x):
    @types(int)
    def decr_(y):
        y = y-1
    x = x + 1

def helloworld():
    print('hello world')

@types(int)
def incr(x):
    x = x + 1

@types(int)
def decr(x):
    y = x - 1
    return y

@types('int[:]')
def incr_array(x):
    x = x + 1

y_=[1,2,3]

@types('[int]')
def decr_array(x):
    y_[1] = 6
    z = y_
    t = y_+x

    return t

@types(int,int,int)
def f1(x, n=2, m=3):
    y = x - n*m
    return y

@types(int,int)
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
