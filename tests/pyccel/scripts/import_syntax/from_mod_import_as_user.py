# pylint: disable=missing-function-docstring, missing-module-docstring
from user_mod import user_func as f

def fun(xi1 : 'double', xi2 : 'double', xi3 : 'double'):
    return f(xi1, xi2, xi3)

if __name__ == '__main__':
    print(fun(1.0,2.0,3.0))
