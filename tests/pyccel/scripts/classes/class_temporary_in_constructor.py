import numpy as np

class MyClass1:
    def __init__(self : 'MyClass1', param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        self.param2 = param2
        print("MyClass1 Object created!")

class MyClass2:
    def __init__(self : 'MyClass2', param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        obj1 = MyClass1(param1, param2)
        self.param2 = sum(obj1.param2)
        print("MyClass2 Object created!")

if __name__ == '__main__':
    p = np.ones(4)
    obj = MyClass2(1, p)
    p = np.array([1., 2., 3., 4.])
    print(obj.param1, obj.param2)
