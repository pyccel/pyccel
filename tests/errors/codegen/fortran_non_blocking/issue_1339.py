# Fortran cannot yet handle a templated function returning either a scalar or an array. If you are using the terminal interface, please pass --language c, if you are using the interactive interfaces epyccel or lambdify, please pass language='c'. See https://github.com/pyccel/pyccel/issues/1339 to monitor the advancement of this issue.
# pylint: disable=missing-function-docstring, missing-module-docstring

def f(a : 'int | int[:]'):
    return a+3
