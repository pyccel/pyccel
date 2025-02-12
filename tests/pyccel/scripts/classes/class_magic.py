# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring

class A:
    def __init__(self, x : int):
        self.x = x

    def __add__(self, other : int):
        return A(self.x+other)

    def __mul__(self, other : int):
        return A(self.x*other)

    def __radd__(self, other : int):
        return A(self.x+other)

    def __iadd__(self, other : int):
        self.x += other
        return self

    def __contains__(self, other : int):
        return self.x == other

    def __len__(self):
        return 2

if __name__ == '__main__':
    my_a = A(4)
    left_add = my_a + 5
    right_add = 7 + my_a
    left_mul = my_a * 2

    print(my_a.x)
    print(left_add.x)
    print(right_add.x)
    print(left_mul.x)

    my_a += 6

    print(my_a.x)

    my_a *= 3

    print(my_a.x)

    print(3 in my_a)
    print(30 in my_a)

    print(len(my_a))
