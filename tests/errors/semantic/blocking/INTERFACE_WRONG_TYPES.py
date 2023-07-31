# pylint: disable=missing-function-docstring, missing-module-docstring

@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
def multi_tmplt_1(x : 'z', y : 'z', z : 'y'):
    """Tests Interfaces"""
    return x + y + z

def tst_multi_tmplt_1():
    """Tests call of the above function"""
    x = multi_tmplt_1(5, 5.3, 7)
    return x
