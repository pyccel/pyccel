#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Handling the transitions between python code and C code using (C Api).
"""

from .datatypes         import (NativeInteger, NativeFloat, NativeComplex,
                                NativeBool, NativeGeneric, NativeVoid)

from .cwrapper          import PyccelPyObject, PyccelPyListObject #PyccelPyArrayObject

from .core              import FunctionDef, FunctionCall

from .internals         import get_final_precision

from .literals          import LiteralInteger

from .operators         import PyccelNot, PyccelEq

from .variable          import Variable

from ..errors.errors   import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

__all__ = (
    #------- CAST FUNCTIONS ------
    'unwrap_list',
    #-------CHECK FUNCTIONS ------
    'list_checker',
    #-------HELPERS ------
    # 'array_get_dim', might need similar functions
    # 'array_get_data',
)

#-------------------------------------------------------------------
#                      Numpy functions
#-------------------------------------------------------------------


unwrap_list = FunctionDef(name      = 'unwrap_list',
                             body      = [],
                             arguments = [Variable(dtype=PyccelPyListObject(), name = 'py_o', memory_handling='alias')],
                             results   = [Variable(dtype=PyccelPyListObject(), name = 'c_o', memory_handling='alias')])

wrap_list = FunctionDef(name      = 'wrap_list',
                             body      = [],
                             arguments = [Variable(dtype=PyccelPyListObject(), name = 'c_o', memory_handling='alias')],
                             results   = [Variable(dtype=PyccelPyListObject(), name = 'py_o', memory_handling='alias')])

pylist_check = FunctionDef(
                name      = 'pylist_check',
                arguments = [
                        Variable(name = 'list', dtype = PyccelPyListObject(), memory_handling='alias'),
                        Variable(name = 'dtype', dtype = NativeInteger())
                    ],
                body      = [],
                results   = [Variable(name = 'b', dtype = NativeBool())])

lst_dtype_registry = {('bool',4)        : 1,
                        ('int',1)       : 2,
                        ('int',2)       : 3,
                        ('int',4)       : 4,
                        ('int',8)       : 5,
                        ('int',16)      : 6,
                        ('float',4)     : 7,
                        ('float',8)     : 8,
                        ('float',16)    : 9,
                        ('complex',4)   : 10,
                        ('complex',8)   : 11,
                        ('list', 4)     : 12}

def find_in_lst_dtype_registry(var):

    dtype = str(var.dtype)
    prec  = get_final_precision(var)
    try :
        return lst_dtype_registry[(dtype, prec)]
    except KeyError:
        return errors.report(PYCCEL_RESTRICTION_TODO,
                symbol = "{}[kind = {}]".format(dtype, prec),
                severity='fatal')


def list_checker(py_variable, c_variable, language):

    type_ref = find_in_lst_dtype_registry(c_variable)

    check = PyccelNot(FunctionCall(pylist_check, [py_variable, type_ref]))

    return check
