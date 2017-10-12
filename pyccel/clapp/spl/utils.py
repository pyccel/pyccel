# coding: utf-8

import numpy as np

from pyccel.clapp.spl import SPL_EvalBasisFunsDers

# ...
def spl_definitions(namespace, declarations):
    """Adds SPL functions and constants to the namespace

    namespace: dict
        dictorionary containing all declared variables/functions/classes.

    declarations: dict
        dictorionary containing all declarations.
    """
    funcs = ['EvalBasisFunsDers']
    for f in funcs:
        atom = eval('SPL_{0}'.format(f))
        namespace[f] = atom()

    return namespace, declarations
# ...
