# pylint: disable=missing-function-docstring, missing-module-docstring
from module_init2 import x

def f():
    def module_init2__init():
        print(123)
    def module_init2__free():
        print(456)
    module_init2__init()
    print(x)
    module_init2__free()

if __name__ == '__main__':
    f()
