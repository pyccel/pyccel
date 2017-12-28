# -*- coding: utf-8 -*-
#

from sympy import symbols, Lambda

from pyccel.symbolic.gelato import glt_symbol

def test_1():
    # ... a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16, 16], \
                      "degrees": [3, 3, 3]}
    # ...

    # ... create a glt symbol from a string without evaluation
    expr = "Ni * Nj + Ni_x * Nj_x + Ni_y * Nj_y + Ni_z * Nj_z"
    expr = glt_symbol(expr, \
                      dim=3, \
                      discretization=discretization, \
                      evaluate=False)
    # ...
    print expr

def test_2():
    # ... a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16, 16], \
                      "degrees": [3, 3, 3]}
    # ...

    # ...
    u, v = symbols('u v')
    f = Lambda((u,v), u*v)
    # ...

    # ... create a glt symbol from a string without evaluation
    expr = glt_symbol(f, \
                      dim=3, \
                      discretization=discretization, \
                      evaluate=False)
    # ...

    print expr

test_1()
test_2()
