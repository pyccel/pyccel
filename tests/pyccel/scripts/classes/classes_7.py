# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class A:
    def __init__(self : 'A', x : int):
        self.x = x

    def update(self : 'A', x : int):
        self.x = x

def get_A():
    return A(4)

def get_x_from_A(a : 'A' = None):
    if a is not None:
        return a.x
    else:
        return 5

if __name__ == '__main__':
    print(get_A().x)
    print(get_x_from_A())
    print(get_x_from_A(get_A()))
    x = get_A()
    x.update(10)
    print(get_x_from_A(x))
