# pylint: disable=missing-function-docstring, missing-module-docstring

awkward_names = 4

def function():
    double = 3.0
    return double


def pure():
    return 1


a = 3.5
A = 5.9

def allocate(void : int):
    return void + 1
