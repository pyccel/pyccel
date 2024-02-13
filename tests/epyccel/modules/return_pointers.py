# pylint: disable=missing-function-docstring, missing-module-docstring

def return_ambiguous_pointer_to_argument(x: 'int[:]'):
    y = x
    w = y
    return w
