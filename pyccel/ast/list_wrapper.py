#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Handling the transitions between python code and C code using (C Api).
"""

import numpy as np ##REMOVE

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
                        Variable(name = 'l', dtype = PyccelPyListObject(), memory_handling='alias'),
                        Variable(name = 'dtype', dtype = NativeInteger())
                    ],
                body      = [],
                results   = [Variable(name = 'b', dtype = NativeBool())])

no_type_check           = Variable(dtype=NativeInteger(),  name = 'NO_TYPE_CHECK')

list_get_type           = FunctionDef()

# https://numpy.org/doc/stable/reference/c-api/dtype.html
numpy_bool_type         = Variable(dtype=NativeInteger(),  name = 'NPY_BOOL', precision = 4)
numpy_byte_type         = Variable(dtype=NativeInteger(),  name = 'NPY_BYTE', precision = 4)
numpy_ubyte_type        = Variable(dtype=NativeInteger(),  name = 'NPY_UBYTE', precision = 4)
numpy_short_type        = Variable(dtype=NativeInteger(),  name = 'NPY_SHORT', precision = 4)
numpy_ushort_type       = Variable(dtype=NativeInteger(),  name = 'NPY_USHORT', precision = 4)
numpy_int_type          = Variable(dtype=NativeInteger(),  name = 'NPY_INT32', precision = 4)
numpy_uint_type         = Variable(dtype=NativeInteger(),  name = 'NPY_UINT', precision = 4)
numpy_long_type         = Variable(dtype=NativeInteger(),  name = 'NPY_LONG', precision = 4)
numpy_ulong_type        = Variable(dtype=NativeInteger(),  name = 'NPY_ULONG', precision = 4)
numpy_longlong_type     = Variable(dtype=NativeInteger(),  name = 'NPY_INT64', precision = 4)
numpy_ulonglong_type    = Variable(dtype=NativeInteger(),  name = 'NPY_ULONGLONG', precision = 4)
numpy_float_type        = Variable(dtype=NativeInteger(),  name = 'NPY_FLOAT', precision = 4)
numpy_double_type       = Variable(dtype=NativeInteger(),  name = 'NPY_DOUBLE', precision = 4)
numpy_longdouble_type   = Variable(dtype=NativeInteger(),  name = 'NPY_LONGDOUBLE', precision = 4)
numpy_cfloat_type       = Variable(dtype=NativeInteger(),  name = 'NPY_CFLOAT', precision = 4)
numpy_cdouble_type      = Variable(dtype=NativeInteger(),  name = 'NPY_CDOUBLE', precision = 4)
numpy_clongdouble_type  = Variable(dtype=NativeInteger(),  name = 'NPY_CLONGDOUBLE', precision = 4)

numpy_num_to_type = {0 : numpy_bool_type,
                     1 : numpy_byte_type,
                     2 : numpy_ubyte_type,
                     3 : numpy_short_type,
                     4 : numpy_ushort_type,
                     5 : numpy_int_type,
                     6 : numpy_uint_type,
                     7 : numpy_long_type,
                     8 : numpy_ulong_type,
                     9 : numpy_longlong_type,
                    10 : numpy_ulonglong_type,
                    11 : numpy_float_type,
                    12 : numpy_double_type,
                    13 : numpy_longdouble_type,
                    14 : numpy_cfloat_type,
                    15 : numpy_cdouble_type,
                    16 : numpy_clongdouble_type}

# This dictionary is required as the precision does not line up with the expected type on windows
numpy_int_type_precision_map = {
        1 : np.dtype(np.int8).num,
        2 : np.dtype(np.int16).num,
        4 : np.dtype(np.int32).num,
        8 : np.dtype(np.int64).num}

numpy_dtype_registry = {('bool',4)     : numpy_bool_type,
                        ('int',1)      : numpy_num_to_type[numpy_int_type_precision_map[1]],
                        ('int',2)      : numpy_num_to_type[numpy_int_type_precision_map[2]],
                        ('int',4)      : numpy_num_to_type[numpy_int_type_precision_map[4]],
                        ('int',8)      : numpy_num_to_type[numpy_int_type_precision_map[8]],
                        ('int',16)     : numpy_longlong_type,
                        ('float',4)    : numpy_float_type,
                        ('float',8)    : numpy_double_type,
                        ('float',16)   : numpy_longdouble_type,
                        ('complex',4)  : numpy_cfloat_type,
                        ('complex',8)  : numpy_cdouble_type,
                        ('complex',16) : numpy_clongdouble_type}

# Needed to check for numpy arguments type
Numpy_Bool_ref       = Variable(dtype=NativeVoid(),  name = 'Bool')
Numpy_Int8_ref       = Variable(dtype=NativeVoid(),  name = 'Int8')
Numpy_Int16_ref      = Variable(dtype=NativeVoid(),  name = 'Int16')
Numpy_Int32_ref      = Variable(dtype=NativeVoid(),  name = 'Int32')
Numpy_Int64_ref      = Variable(dtype=NativeVoid(),  name = 'Int64')
Numpy_Float_ref      = Variable(dtype=NativeVoid(),  name = 'Float32')
Numpy_Double_ref     = Variable(dtype=NativeVoid(),  name = 'Float64')
Numpy_Complex64_ref  = Variable(dtype=NativeVoid(),  name = 'Complex64')
Numpy_Complex128_ref = Variable(dtype=NativeVoid(),  name = 'Complex128')

numpy_type_check_registry = {
    (NativeInteger(), 4)       : Numpy_Int32_ref,
    (NativeInteger(), 8)       : Numpy_Int64_ref,
    (NativeInteger(), 2)       : Numpy_Int16_ref,
    (NativeInteger(), 1)       : Numpy_Int8_ref,
    (NativeFloat(), 8)         : Numpy_Double_ref,
    (NativeFloat(), 4)         : Numpy_Float_ref,
    (NativeComplex(), 4)       : Numpy_Complex64_ref,
    (NativeComplex(), 8)       : Numpy_Complex128_ref,
    (NativeBool(), 4)          : Numpy_Bool_ref
}

# helpers
def find_in_numpy_dtype_registry(var):

    dtype = str(var.dtype)
    prec  = get_final_precision(var)
    try :
        return numpy_dtype_registry[(dtype, prec)]
    except KeyError:
        return errors.report(PYCCEL_RESTRICTION_TODO,
                symbol = "{}[kind = {}]".format(dtype, prec),
                severity='fatal')


def list_checker(py_variable, c_variable, type_check_needed, language):

    rank     = c_variable.rank
    type_ref = no_type_check

    if type_check_needed:
        type_ref = find_in_numpy_dtype_registry(c_variable)

    check = PyccelNot(FunctionCall(pylist_check, [py_variable, type_ref]))

    return check

