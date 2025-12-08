# pylint: disable=missing-function-docstring, missing-module-docstring

def f(*a : int, **b : int):
    print(len(a))
    for ai in a:
        print(ai)
    for bk, bv in b.items():
        print(bk, bv)

if __name__ == '__main__':
    f(1, 2, 3, c = 4, b = 5)
    f(1, 2, 3, 4, a = 1)
