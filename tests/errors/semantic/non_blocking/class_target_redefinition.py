# Attempt to reallocate an object which is being aliased by another variable
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np

class MyClass:
    def __init__(self : 'MyClass', param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        self.param2 = param2
        print("MyClass Object created!")


if __name__ == '__main__':
    p = np.ones(4)
    obj = MyClass(1, p)
    p = np.array([1., 2., 3., 4.])
    print(obj.param1, obj.param2)
