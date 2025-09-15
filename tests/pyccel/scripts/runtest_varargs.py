# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import private

@private
def f(*a : int):
    print(len(a))
    for ai in a:
        print(ai)

@private
def g(a : int, b : float, *c : int):
    print(a)
    print(int(b))
    f(*c)

def main():
    f(1,2,3)
    f(1,2,3,4)

    tup1 = (3, 4.50)
    tup2 = (3,4)
    g(*tup1, *tup2)
    g(*tup1)

if __name__ == '__main__':
    main()
