# coding: utf-8

from pyccel.parser.utilities import find_imports

#stmts = 'from pyccel.stdlib import *'

def test_numpy():
    d = find_imports(stmts='from numpy import zeros')
    assert(d['numpy'] == ['zeros'])

    d = find_imports(stmts='from numpy import zeros, dot')
    assert(d['numpy'] == ['zeros', 'dot'])

#    stmts = 'from numpy import *'
#    d = find_imports(stmts=stmts)

test_numpy()
