# pylint: disable=missing-function-docstring, missing-module-docstring

def fun(xi1 : 'double', xi2 : 'double', xi3 : 'double'):
    import user_mod
    return user_mod.user_func(xi1, xi2, xi3)

if __name__ == '__main__':
    print(fun(1.0,2.0,3.0))
