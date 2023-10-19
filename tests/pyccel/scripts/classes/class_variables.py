# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    y : int = 5
    def __init__(self : 'A'):
        self.x = 3

    def get_4(self : 'A'):
        return 4

if __name__ == '__main__':
    myA : 'A' = A()

    print(myA.x)
