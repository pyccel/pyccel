# pylint: disable=missing-function-docstring, missing-module-docstring

@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
def multi_tmplt_1(x : 'z', y : 'z', z : 'y'):
    return x + y + z

@template('z', types=['int'])
@template('y', types=['int', 'real'])
def multi_tmplt_2(y : 'z', z : 'y'):
    return y + z

def tst_multi_tmplt_2():
    x = multi_tmplt_2(5.4, 5.4)
    return x
