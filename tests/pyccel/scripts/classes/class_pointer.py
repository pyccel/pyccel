# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

class A:
    def __init__(self, a : int):
        self._a = a

    def get_a(self):
        return self._a

    def set_a(self, a : int):
        self._a = a

if __name__ == '__main__':
    my_a = A(3)
    my_a_ptr = my_a
    print(my_a.get_a())
    print(my_a_ptr.get_a())
    my_a_ptr.set_a(4)
    print(my_a.get_a())
    print(my_a_ptr.get_a())
