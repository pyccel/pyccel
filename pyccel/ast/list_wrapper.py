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

lst_dtype_registry = {('bool',4)        : Variable(dtype=NativeInteger(),  name = 'lst_bool', precision = 4),
                        ('int',1)       : Variable(dtype=NativeInteger(),  name = 'lst_int8', precision = 1),
                        ('int',2)       : Variable(dtype=NativeInteger(),  name = 'lst_int16', precision = 2),
                        ('int',4)       : Variable(dtype=NativeInteger(),  name = 'lst_int32', precision = 4),
                        ('int',8)       : Variable(dtype=NativeInteger(),  name = 'lst_int64', precision = 8),
                        ('int',16)      : Variable(dtype=NativeInteger(),  name = 'lst_long', precision = 16),
                        ('float',4)     : Variable(dtype=NativeInteger(),  name = 'lst_float', precision = 4),
                        ('float',8)     : Variable(dtype=NativeInteger(),  name = 'lst_double', precision = 8),
                        ('complex',4)   : Variable(dtype=NativeInteger(),  name = 'lst_complex', precision = 4),
                        ('complex',8)   : Variable(dtype=NativeInteger(),  name = 'lst_dcomplex0', precision = 8),
                        ('list', 4)     : Variable(dtype=NativeInteger(),  name = 'lst_list', precision = 4)}

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
