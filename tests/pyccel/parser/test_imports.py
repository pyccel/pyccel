# coding: utf-8

from pyccel.parser.utilities import find_imports

#stmts = 'from pyccel.stdlib import *'

def test_numpy():
    d = find_imports(stmts='from numpy import zeros')
    assert(d['numpy'] == ['zeros'])

    d = find_imports(stmts='from numpy import zeros, dot')
    assert(d['numpy'] == ['zeros', 'dot'])

    d = find_imports(stmts='from numpy import *')
    assert(d['numpy'] == ['*'])

test_numpy()
