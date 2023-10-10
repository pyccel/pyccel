# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

class A(object):
    def __init__(self : 'A', x : int):
        self.x = x

    def __del__(self : 'A'):
        pass

    def f(self : 'A'):
        return self.x+2

def get_A():
    a_cls = A(3)
    return a_cls

if __name__ == '__main__':
    b = get_A().x
    c = get_A().f()+3

    print(b)
    print(c)
