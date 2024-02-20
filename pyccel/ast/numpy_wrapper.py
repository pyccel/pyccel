#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Handling the transitions between Python code and C code using (Numpy/C Api).
"""

import numpy as np

from pyccel.utilities.metaclasses import Singleton

from .datatypes         import PythonNativeBool, GenericType, VoidType, FixedSizeType

from .cwrapper          import PyccelPyObject

from .core              import FunctionDef, FunctionCall
from .core              import FunctionDefArgument, FunctionDefResult

from .c_concepts        import CNativeInt

from .literals          import LiteralInteger

from .numpytypes        import NumpyInt8Type, NumpyInt16Type, NumpyInt32Type, NumpyInt64Type
from .numpytypes        import NumpyFloat32Type, NumpyFloat64Type, NumpyFloat128Type
from .numpytypes        import NumpyComplex64Type, NumpyComplex128Type, NumpyComplex256Type
from .numpytypes        import NumpyNDArrayType

from .variable          import Variable

from ..errors.errors   import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

__all__ = (
    #--------- DATATYPES ---------
    'PyccelPyArrayObject',
    #------- CAST FUNCTIONS ------
    'pyarray_to_ndarray',
    #-------CHECK FUNCTIONS ------
    'array_type_check',
    'scalar_type_check',
    #-------HELPERS ------
    'array_get_dim',
    'array_get_data',
    'array_get_c_step',
    'array_get_f_step',
    'PyArray_SetBaseObject',
    #-------OTHERS--------
    'get_numpy_max_acceptable_version_file',
)

class PyccelPyArrayObject(FixedSizeType):
    """
    Datatype representing a `PyArrayObject`.

    Datatype representing a `PyArrayObject` which is the
    class used to hold NumPy array objects in Python.
    """
    __slots__ = ()
    _name = 'PyArrayObject'

#-------------------------------------------------------------------
#                      Numpy functions
#-------------------------------------------------------------------

def get_numpy_max_acceptable_version_file():
    """
    Get the macro specifying the last acceptable numpy version. If numpy is more
    recent than this then deprecation warnings are shown.

    The last acceptable numpy version is 1.19. If the current version is older
    than this then the last acceptable numpy version is the current version
    """
    numpy_max_acceptable_version = [1, 19]
    numpy_current_version = [int(v) for v in np.version.version.split('.')[:2]]
    numpy_api_acceptable_version = min(numpy_max_acceptable_version, numpy_current_version)
    numpy_api_macro = '# define NPY_NO_DEPRECATED_API NPY_{}_{}_API_VERSION\n'.format(
        *numpy_api_acceptable_version)

    return '#ifndef NPY_NO_DEPRECATED_API\n'+ \
            numpy_api_macro+\
           '#endif'

PyArray_Check = FunctionDef(name      = 'PyArray_Check',
                            body      = [],
                            arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o'))],
                            results   = [FunctionDefResult(Variable(dtype=PythonNativeBool(), name='b'))])

# numpy array to c ndarray : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
pyarray_to_ndarray = FunctionDef(
                name      = 'pyarray_to_ndarray',
                arguments = [FunctionDefArgument(Variable(name = 'a', dtype = PyccelPyObject(), memory_handling = 'alias'))],
                body      = [],
                results   = [FunctionDefResult(Variable(name = 'array', dtype = NumpyNDArrayType(GenericType())))])

# numpy array check elements : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
pyarray_check = FunctionDef(
                name      = 'pyarray_check',
                arguments = [
                        FunctionDefArgument(Variable(name = 'a', dtype = PyccelPyObject(), memory_handling='alias')),
                        FunctionDefArgument(Variable(name = 'dtype', dtype = CNativeInt())),
                        FunctionDefArgument(Variable(name = 'rank', dtype = CNativeInt())),
                        FunctionDefArgument(Variable(name = 'flag', dtype = CNativeInt()))
                    ],
                body      = [],
                results   = [FunctionDefResult(Variable(name = 'b', dtype = PythonNativeBool()))])

is_numpy_array = FunctionDef(
                name      = 'is_numpy_array',
                arguments = [
                        FunctionDefArgument(Variable(name = 'a', dtype = PyccelPyObject(), memory_handling='alias')),
                        FunctionDefArgument(Variable(name = 'dtype', dtype = CNativeInt())),
                        FunctionDefArgument(Variable(name = 'rank', dtype = CNativeInt())),
                        FunctionDefArgument(Variable(name = 'flag', dtype = CNativeInt()))
                    ],
                body      = [],
                results   = [FunctionDefResult(Variable(name = 'b', dtype = PythonNativeBool()))])

# Return the shape of the n-th dimension : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
array_get_dim  = FunctionDef(name    = 'nd_ndim',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(dtype=VoidType(), name = 'o', is_optional = True)),
                                        FunctionDefArgument(Variable(dtype=CNativeInt(), name = 'idx'))],
                           results   = [FunctionDefResult(Variable(dtype=CNativeInt(), name = 'd'))])

# Return the stride of the n-th dimension : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
array_get_c_step = FunctionDef(name    = 'nd_nstep_C',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(dtype=VoidType(), name = 'o', is_optional = True)),
                                        FunctionDefArgument(Variable(dtype=CNativeInt(), name = 'idx'))],
                           results   = [FunctionDefResult(Variable(dtype=CNativeInt(), name = 'd'))])
array_get_f_step = FunctionDef(name    = 'nd_nstep_F',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(dtype=VoidType(), name = 'o', is_optional = True)),
                                        FunctionDefArgument(Variable(dtype=CNativeInt(), name = 'idx'))],
                           results   = [FunctionDefResult(Variable(dtype=CNativeInt(), name = 'd'))])

# Return the data of ndarray : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
array_get_data  = FunctionDef(name   = 'nd_data',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(dtype=VoidType(), name = 'o', is_optional=True))],
                           results   = [FunctionDefResult(Variable(dtype=VoidType(), name = 'v', memory_handling='alias', rank = 1, class_type = VoidType()))])

PyArray_SetBaseObject = FunctionDef(name   = 'PyArray_SetBaseObject',
                                    body      = [],
                                    arguments = [FunctionDefArgument(Variable(dtype=PyccelPyArrayObject(), name = 'arr', memory_handling='alias')),
                                                 FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'obj', memory_handling='alias'))],
                                    results   = [FunctionDefResult(Variable(dtype=CNativeInt(), name = 'd'))])

import_array = FunctionDef('import_array', (), (), ())

# Basic Array Flags
# https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_OWNDATA
numpy_flag_own_data     = Variable(dtype=CNativeInt(),  name = 'NPY_ARRAY_OWNDATA')
# https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_C_CONTIGUOUS
numpy_flag_c_contig     = Variable(dtype=CNativeInt(),  name = 'NPY_ARRAY_C_CONTIGUOUS')
# https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_F_CONTIGUOUS
numpy_flag_f_contig     = Variable(dtype=CNativeInt(),  name = 'NPY_ARRAY_F_CONTIGUOUS')

# Custom Array Flags defined in pyccel/stdlib/cwrapper/cwrapper_ndarrays.h
no_type_check           = Variable(dtype=CNativeInt(),  name = 'NO_TYPE_CHECK')
no_order_check          = Variable(dtype=CNativeInt(),  name = 'NO_ORDER_CHECK')

# https://numpy.org/doc/stable/reference/c-api/dtype.html
numpy_bool_type         = Variable(dtype=CNativeInt(),  name = 'NPY_BOOL')
numpy_byte_type         = Variable(dtype=CNativeInt(),  name = 'NPY_BYTE')
numpy_ubyte_type        = Variable(dtype=CNativeInt(),  name = 'NPY_UBYTE')
numpy_short_type        = Variable(dtype=CNativeInt(),  name = 'NPY_SHORT')
numpy_ushort_type       = Variable(dtype=CNativeInt(),  name = 'NPY_USHORT')
numpy_int_type          = Variable(dtype=CNativeInt(),  name = 'NPY_INT32')
numpy_uint_type         = Variable(dtype=CNativeInt(),  name = 'NPY_UINT')
numpy_long_type         = Variable(dtype=CNativeInt(),  name = 'NPY_LONG')
numpy_ulong_type        = Variable(dtype=CNativeInt(),  name = 'NPY_ULONG')
numpy_longlong_type     = Variable(dtype=CNativeInt(),  name = 'NPY_INT64')
numpy_ulonglong_type    = Variable(dtype=CNativeInt(),  name = 'NPY_ULONGLONG')
numpy_float_type        = Variable(dtype=CNativeInt(),  name = 'NPY_FLOAT')
numpy_double_type       = Variable(dtype=CNativeInt(),  name = 'NPY_DOUBLE')
numpy_longdouble_type   = Variable(dtype=CNativeInt(),  name = 'NPY_LONGDOUBLE')
numpy_cfloat_type       = Variable(dtype=CNativeInt(),  name = 'NPY_CFLOAT')
numpy_cdouble_type      = Variable(dtype=CNativeInt(),  name = 'NPY_CDOUBLE')
numpy_clongdouble_type  = Variable(dtype=CNativeInt(),  name = 'NPY_CLONGDOUBLE')

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

numpy_dtype_registry = {NumpyInt8Type()       : numpy_num_to_type[numpy_int_type_precision_map[1]],
                        NumpyInt16Type()      : numpy_num_to_type[numpy_int_type_precision_map[2]],
                        NumpyInt32Type()      : numpy_num_to_type[numpy_int_type_precision_map[4]],
                        NumpyInt64Type()      : numpy_num_to_type[numpy_int_type_precision_map[8]],
                        NumpyFloat32Type()    : numpy_float_type,
                        NumpyFloat64Type()    : numpy_double_type,
                        NumpyFloat128Type()   : numpy_longdouble_type,
                        NumpyComplex64Type()  : numpy_cfloat_type,
                        NumpyComplex128Type() : numpy_cdouble_type,
                        NumpyComplex256Type() : numpy_clongdouble_type}

# Needed to check for numpy arguments type
Numpy_Bool_ref       = Variable(dtype=VoidType(),  name = 'Bool')
Numpy_Int8_ref       = Variable(dtype=VoidType(),  name = 'Int8')
Numpy_Int16_ref      = Variable(dtype=VoidType(),  name = 'Int16')
Numpy_Int32_ref      = Variable(dtype=VoidType(),  name = 'Int32')
Numpy_Int64_ref      = Variable(dtype=VoidType(),  name = 'Int64')
Numpy_Float_ref      = Variable(dtype=VoidType(),  name = 'Float32')
Numpy_Double_ref     = Variable(dtype=VoidType(),  name = 'Float64')
Numpy_Complex64_ref  = Variable(dtype=VoidType(),  name = 'Complex64')
Numpy_Complex128_ref = Variable(dtype=VoidType(),  name = 'Complex128')

numpy_type_check_registry = {
    NumpyInt8Type()       : Numpy_Int8_ref,
    NumpyInt16Type()      : Numpy_Int16_ref,
    NumpyInt32Type()      : Numpy_Int32_ref,
    NumpyInt64Type()      : Numpy_Int64_ref,
    NumpyFloat32Type()    : Numpy_Float_ref,
    NumpyFloat64Type()    : Numpy_Double_ref,
    NumpyComplex64Type()  : Numpy_Complex64_ref,
    NumpyComplex128Type() : Numpy_Complex128_ref,
}

# helpers
def array_type_check(py_variable, c_variable, raise_error):
    """
    Return the code which checks if the array has the expected type.

    Returns the code which checks if the array has the expected rank,
    datatype, and order. These are determined from the
    properties of the `c_variable` argument.

    Parameters
    ----------
    py_variable : Variable
            A variable containing the Python object passed into the wrapper.
    c_variable : Variable
            A variable containing the basic C object which will store the array.
    raise_error : bool
            Indicates whether an error should be raised if the type does not match.

    Returns
    -------
    FunctionCall
            The code necessary to validate the provided array.
    """
    rank     = c_variable.rank
    flag     = no_order_check
    try :
        type_ref = numpy_dtype_registry[var.dtype]
    except KeyError:
        return errors.report(PYCCEL_RESTRICTION_TODO,
                symbol = dtype,
                severity='fatal')
    # order flag
    if rank > 1:
        if c_variable.order == 'F':
            flag = numpy_flag_f_contig
        else:
            flag = numpy_flag_c_contig

    if raise_error:
        return FunctionCall(pyarray_check, [py_variable, type_ref, LiteralInteger(rank), flag])
    else:
        return FunctionCall(is_numpy_array, [py_variable, type_ref, LiteralInteger(rank), flag])


def scalar_type_check(py_variable, c_variable):
    """
    Create a FunctionCall to check the type of a Python object.

    Create a FunctionCall object representing a call to a function which
    is responsible for checking if the Python object passed as an argument
    has a type matching that of the provided C object.

    Parameters
    ----------
    py_variable : Variable
        The Python argument of the check function.

    c_variable : Variable
        The variable needed for the generation of the type check.

    Returns
    -------
    FunctionCall
        The FunctionCall which checks the type.
    """
    try :
        check_numpy_ref = numpy_type_check_registry[c_variable.dtype]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_variable.dtype,severity='fatal')

    check_numpy_func = FunctionDef(name = 'PyArray_IsScalar',
                              body      = [],
                              arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias')),
                                           FunctionDefArgument(check_numpy_ref)],
                              results   = [FunctionDefResult(Variable(dtype=PythonNativeBool(), name = 'r'))])

    return FunctionCall(check_numpy_func, [py_variable, check_numpy_ref])
