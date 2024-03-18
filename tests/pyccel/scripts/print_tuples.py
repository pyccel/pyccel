# pylint: disable=missing-function-docstring, missing-module-docstring
# ------------------------------- Strings ------------------------------------

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
