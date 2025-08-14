# Variable my_array goes out of scope but may be the target of a pointer which is still required
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np

class MyClass:
    def __init__(self, param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        self.param2 = param2
        print("MyClass Object created!")

def get_MyClass():
    my_array = np.ones(4)
    my_class = MyClass(2, my_array)
    my_array[2] = 4.0 # Must also modify the array in my_class
    return my_class # Pyccel automatically deallocates my_array as it goes out of scope, but my_class is still in scope

if __name__ == '__main__':
    obj = get_MyClass()
    print(obj.param1, obj.param2)
