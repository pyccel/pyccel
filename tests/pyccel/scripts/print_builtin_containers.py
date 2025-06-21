# pylint: disable=missing-module-docstring,missing-function-docstring

def f():
    return 1, True


if __name__ == '__main__':
    print(())
    print((1,))
    print((1,2,3))
    print(((1,2),3))
    print(((1,2),(3,)))
    print((((1,),2),(3,)))
    print((1, True))
    print((1, False), sep=",")
    print((1, True), end="!\n")
    print(f())

    print([1,2,3])
    a = [4,5,6]
    print(a)
    b : list[complex] = []
    print(b)
    c = [1.0, 2.0, 3.0]
    print(c)

    print({1,2,3})

