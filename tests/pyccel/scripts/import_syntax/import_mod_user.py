# pylint: disable=missing-function-docstring, missing-module-docstring
import user_mod

def fun(xi1 : 'float', xi2 : 'float', xi3 : 'float'):
    return user_mod.user_func(xi1, xi2, xi3)

if __name__ == '__main__':
    print(fun(1.0,2.0,3.0))
