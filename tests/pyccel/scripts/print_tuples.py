# pylint: disable=missing-function-docstring, missing-module-docstring/
# ------------------------------- Strings ------------------------------------
if __name__ == '__main__':
    print(())
    print((1,))
    print((1,2,3))
    print(((1,2),3))
    print(((1,2),(3,)))
    print((((1,),2),(3,)))
    print((1, True))
    print((1, True), sep=",")
    print((1, True), end="!\n")
