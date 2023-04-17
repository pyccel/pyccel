# pylint: disable=missing-function-docstring, missing-module-docstring
@types('complex', 'complex')
def compare_isnot_complex(a, b):
    return a is not b
