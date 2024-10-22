# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring

class A:
    def __init__(self, x : int):
        self.x = x

    def __add__(self, other : int):
        return A(self.x+other)

    def __radd__(self, other : int):
        return A(self.x+other)

if __name__ == '__main__':
    my_a = A(4)
    left_add = my_a + 5
    right_add = 7 + my_a

    print(my_a.x)
    print(left_add.x)
    print(right_add.x)
