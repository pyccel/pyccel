# pylint: disable=missing-function-docstring, missing-module-docstring/
@types('int', 'int')
def compare_isnot_int_int(a, b):
    return a is not b
