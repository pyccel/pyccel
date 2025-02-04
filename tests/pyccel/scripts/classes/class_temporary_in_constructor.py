# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np

class MyClass1:
    def __init__(self, param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        self.param2 = param2

class MyClass2:
    def __init__(self, param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        obj1 = MyClass1(param1, param2)
        self.param2 = np.sum(obj1.param2)

if __name__ == '__main__':
    p = np.ones(4)
    obj = MyClass2(1, p)
    p = np.array([1., 2., 3., 4.])
    print(obj.param1, obj.param2)
