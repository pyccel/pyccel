# pylint: disable=missing-function-docstring, missing-module-docstring

def fun(xi1 : 'float', xi2 : 'float', xi3 : 'float'):
    import user_mod as u
    return u.user_func(xi1, xi2, xi3)

if __name__ == '__main__':
    print(fun(1.0,2.0,3.0))

