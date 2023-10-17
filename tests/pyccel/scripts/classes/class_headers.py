# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self : 'A'):
        self.x = 3

if __name__ == '__main__':
    #$ header variable myA A
    myA = A()

    print(myA.x)
