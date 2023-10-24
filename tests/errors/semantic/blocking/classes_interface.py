# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class MyClass:
    def __init__(self : 'MyClass', param1 : 'int'):
        self.param1 = param1
        print("MyClass Object created!")

    def help(self : 'MyClass', param1 : 'int|float'):
        self.param1 = param1
