# coding: utf-8

import numpy as np

from pyccel.clapp.plaf import (Matrix_dns, Matrix_csr, Matrix_csc, \
                               Matrix_bnd, Matrix_coo)

from pyccel.types.ast import DataTypeFactory

# ...
def plaf_definitions(namespace, declarations, cls_constructs):
    """Adds PLAF functions, classes and constants to the namespace

    namespace: dict
        dictionary containing all declared variables/functions/classes.

    declarations: dict
        dictionary containing all declarations.

    cls_constructs: dict
        dictionary of datatypes of classes using DatatypeFactory

    """
#    func_defs = []
#    for i in func_defs:
#        atom = eval('{0}'.format(i))
#        namespace[f] = atom()

    classes = ['Matrix_dns', 'Matrix_csr', 'Matrix_csc', \
               'Matrix_bnd', 'Matrix_coo']
    for i in classes:
        name = 'plf_t_{0}'.format(i.lower())
        cls_constructs[name] = DataTypeFactory(i, ("_name"))

    return namespace, declarations, cls_constructs
# ...
