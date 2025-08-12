# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np
from pyccel.decorators import inline

class MyClass:
    def __init__(self, param1 : 'int', param2 : 'int'):
        self.param1 = param1
        self.param2 = np.ones(param2, dtype=int)
        print(12345)

    @inline
    def get_param(self):
        print(self.param1, self.param2)

class MyClass1:
    def __init__(self):
        print(54321)

    def Method1(self, param1 : MyClass):
        self.param = param1 #pylint: disable=attribute-defined-outside-init

    def Method2(self):
        return MyClass(2, 4)

if __name__ == '__main__':
    obj = MyClass1()
    obj.Method1(obj.Method2())
    obj.param.get_param()

