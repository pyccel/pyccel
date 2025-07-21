# pylint: disable=missing-function-docstring, missing-module-docstring

def f(*a : int):
    for ai in a:
        print(ai)

if __name__ == '__main__':
    f(1,2,3)
    f(1,2,3,4)
