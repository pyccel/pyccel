from pyccel.decorators import types

@types('double','double','double')
def user_func(x1, x2, x3):
    return x1+x2+x3
