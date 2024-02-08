# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np

class MyClass:
    def __init__(self, param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        self.param2 = np.sum(param2[::2])


if __name__ == '__main__':
    p = np.ones(4)
    obj = MyClass(1, p)
    p = np.array([1., 2., 3., 4.])
    print(obj.param1, obj.param2)
