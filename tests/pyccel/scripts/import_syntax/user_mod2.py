from pyccel.decorators import types

@types('double','double')
def user_func(x1, x2):
    return x1+x2
