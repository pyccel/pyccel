# pylint: disable=missing-function-docstring, missing-module-docstring

def f(*a : int):
    print(len(a))
    for ai in a:
        print(ai)

def g(a : int, b : float, *c : int):
    print(a)
    print(int(b))
    f(*c)

if __name__ == '__main__':
    f(1,2,3)
    f(1,2,3,4)

    tup1 = (3, 4.50)
    tup2 = (3,4)
    g(*tup1, *tup2)
    g(*tup1)
