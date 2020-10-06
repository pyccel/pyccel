# pylint: disable=missing-function-docstring, missing-module-docstring/
@types('float', 'float')
def compare_isnot_float(a, b):
    return a is not b
