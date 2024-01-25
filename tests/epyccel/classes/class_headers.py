# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

#$ header method get_4(A) results(int)

class A:
    def __init__(self : 'A'):
        self.x = 3

    def get_4(self):
        return 4

if __name__ == '__main__':
    #$ header variable myA A
    myA = A()

    print(myA.x)
