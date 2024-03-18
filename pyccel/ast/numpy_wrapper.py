#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Handling the transitions between Python code and C code using (Numpy/C Api).
"""

import numpy as np

from .datatypes         import PythonNativeBool, GenericType, VoidType, FixedSizeType

from .cwrapper          import PyccelPyObject, check_type_registry, c_to_py_registry, pytype_parse_registry

from .core              import FunctionDef
from .core              import FunctionDefArgument, FunctionDefResult

from .c_concepts        import CNativeInt

from .numpytypes        import NumpyInt8Type, NumpyInt16Type, NumpyInt32Type, NumpyInt64Type
from .numpytypes        import NumpyFloat32Type, NumpyFloat64Type, NumpyFloat128Type
from .numpytypes        import NumpyComplex64Type, NumpyComplex128Type, NumpyComplex256Type
from .numpytypes        import NumpyNDArrayType

from .variable          import Variable

from ..errors.errors   import Errors

errors = Errors()

__all__ = (
    #--------- DATATYPES ---------
    'PyccelPyArrayObject',
    #------- CAST FUNCTIONS ------
    'pyarray_to_ndarray',
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
    Get the macro specifying the most recent acceptable NumPy version.

    Get the macro specifying the most recent acceptable NumPy version.
    If NumPy is more recent than this then deprecation warnings are shown.

    The most recent acceptable NumPy version is 1.19. If the current version is older
    than this then the last acceptable NumPy version is the current version.

    Returns
    -------
    str
        A string containing the code which defines the macro.
    """
    numpy_max_acceptable_version = [1, 19]
    numpy_current_version = [int(v) for v in np.version.version.split('.')[:2]]
    numpy_api_acceptable_version = min(numpy_max_acceptable_version, numpy_current_version)
    major, minor = numpy_api_acceptable_version
    numpy_api_macro = f'# define NPY_NO_DEPRECATED_API NPY_{major}_{minor}_API_VERSION\n'

    return '#ifndef NPY_NO_DEPRECATED_API\n'+ \
            numpy_api_macro+\
           '#endif'

PyArray_Check = FunctionDef(name      = 'PyArray_Check',
                            body      = [],
                            arguments = [FunctionDefArgument(Variable(PyccelPyObject(), name = 'o'))],
                            results   = [FunctionDefResult(Variable(PythonNativeBool(), name='b'))])

# NumPy array to c ndarray : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
pyarray_to_ndarray = FunctionDef(
                name      = 'pyarray_to_ndarray',
                arguments = [FunctionDefArgument(Variable(PyccelPyObject(), 'a', memory_handling = 'alias'))],
                body      = [],
                results   = [FunctionDefResult(Variable(NumpyNDArrayType(GenericType()), 'array'))])

# NumPy array check elements : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
pyarray_check = FunctionDef(
                name      = 'pyarray_check',
                arguments = [
                        FunctionDefArgument(Variable(PyccelPyObject(), 'a', memory_handling='alias')),
                        FunctionDefArgument(Variable(CNativeInt(), 'dtype')),
                        FunctionDefArgument(Variable(CNativeInt(), 'rank')),
                        FunctionDefArgument(Variable(CNativeInt(), 'flag'))
                    ],
                body      = [],
                results   = [FunctionDefResult(Variable(PythonNativeBool(), 'b'))])

is_numpy_array = FunctionDef(
                name      = 'is_numpy_array',
                arguments = [
                        FunctionDefArgument(Variable(PyccelPyObject(), 'a', memory_handling='alias')),
                        FunctionDefArgument(Variable(CNativeInt(), 'dtype')),
                        FunctionDefArgument(Variable(CNativeInt(), 'rank')),
                        FunctionDefArgument(Variable(CNativeInt(), 'flag'))
                    ],
                body      = [],
                results   = [FunctionDefResult(Variable(PythonNativeBool(), 'b'))])

# Return the shape of the n-th dimension : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
array_get_dim  = FunctionDef(name    = 'nd_ndim',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(VoidType(), name = 'o', is_optional = True)),
                                        FunctionDefArgument(Variable(CNativeInt(), name = 'idx'))],
                           results   = [FunctionDefResult(Variable(CNativeInt(), name = 'd'))])

# Return the stride of the n-th dimension : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
array_get_c_step = FunctionDef(name    = 'nd_nstep_C',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(VoidType(), name = 'o', is_optional = True)),
                                        FunctionDefArgument(Variable(CNativeInt(), name = 'idx'))],
                           results   = [FunctionDefResult(Variable(CNativeInt(), name = 'd'))])
array_get_f_step = FunctionDef(name    = 'nd_nstep_F',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(VoidType(), name = 'o', is_optional = True)),
                                        FunctionDefArgument(Variable(CNativeInt(), name = 'idx'))],
                           results   = [FunctionDefResult(Variable(CNativeInt(), name = 'd'))])

# Return the data of ndarray : function definition in pyccel/stdlib/cwrapper/cwrapper_ndarrays.c
array_get_data  = FunctionDef(name   = 'nd_data',
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(VoidType(), name = 'o', is_optional=True))],
                           results   = [FunctionDefResult(Variable(VoidType(), name = 'v', memory_handling='alias', rank = 1))])

PyArray_SetBaseObject = FunctionDef(name   = 'PyArray_SetBaseObject',
                                    body      = [],
                                    arguments = [FunctionDefArgument(Variable(PyccelPyArrayObject(), name = 'arr', memory_handling='alias')),
                                                 FunctionDefArgument(Variable(PyccelPyObject(), name = 'obj', memory_handling='alias'))],
                                    results   = [FunctionDefResult(Variable(CNativeInt(), name = 'd'))])

import_array = FunctionDef('import_array', (), (), ())

# Basic Array Flags
# https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_OWNDATA
numpy_flag_own_data     = Variable(CNativeInt(),  name = 'NPY_ARRAY_OWNDATA')
# https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_C_CONTIGUOUS
numpy_flag_c_contig     = Variable(CNativeInt(),  name = 'NPY_ARRAY_C_CONTIGUOUS')
# https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_F_CONTIGUOUS
numpy_flag_f_contig     = Variable(CNativeInt(),  name = 'NPY_ARRAY_F_CONTIGUOUS')

# Custom Array Flags defined in pyccel/stdlib/cwrapper/cwrapper_ndarrays.h
no_type_check           = Variable(CNativeInt(),  name = 'NO_TYPE_CHECK')
no_order_check          = Variable(CNativeInt(),  name = 'NO_ORDER_CHECK')

# https://numpy.org/doc/stable/reference/c-api/dtype.html
numpy_bool_type         = Variable(CNativeInt(),  name = 'NPY_BOOL')
numpy_byte_type         = Variable(CNativeInt(),  name = 'NPY_BYTE')
numpy_ubyte_type        = Variable(CNativeInt(),  name = 'NPY_UBYTE')
numpy_short_type        = Variable(CNativeInt(),  name = 'NPY_SHORT')
numpy_ushort_type       = Variable(CNativeInt(),  name = 'NPY_USHORT')
numpy_int_type          = Variable(CNativeInt(),  name = 'NPY_INT32')
numpy_uint_type         = Variable(CNativeInt(),  name = 'NPY_UINT')
numpy_long_type         = Variable(CNativeInt(),  name = 'NPY_LONG')
numpy_ulong_type        = Variable(CNativeInt(),  name = 'NPY_ULONG')
numpy_longlong_type     = Variable(CNativeInt(),  name = 'NPY_INT64')
numpy_ulonglong_type    = Variable(CNativeInt(),  name = 'NPY_ULONGLONG')
numpy_float_type        = Variable(CNativeInt(),  name = 'NPY_FLOAT')
numpy_double_type       = Variable(CNativeInt(),  name = 'NPY_DOUBLE')
numpy_longdouble_type   = Variable(CNativeInt(),  name = 'NPY_LONGDOUBLE')
numpy_cfloat_type       = Variable(CNativeInt(),  name = 'NPY_CFLOAT')
numpy_cdouble_type      = Variable(CNativeInt(),  name = 'NPY_CDOUBLE')
numpy_clongdouble_type  = Variable(CNativeInt(),  name = 'NPY_CLONGDOUBLE')

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

numpy_dtype_registry = {PythonNativeBool()    : numpy_bool_type,
                        NumpyInt8Type()       : numpy_num_to_type[numpy_int_type_precision_map[1]],
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
check_type_registry.update({
    NumpyInt8Type()       : 'PyIs_Int8',
    NumpyInt16Type()      : 'PyIs_Int16',
    NumpyInt32Type()      : 'PyIs_Int32',
    NumpyInt64Type()      : 'PyIs_Int64',
    NumpyFloat32Type()    : 'PyIs_Float',
    NumpyFloat64Type()    : 'PyIs_Double',
    NumpyComplex64Type()  : 'PyIs_Complex64',
    NumpyComplex128Type() : 'PyIs_Complex128'
    })

c_to_py_registry.update({
    NumpyInt8Type()       : 'Int8_to_NumpyLong',
    NumpyInt16Type()      : 'Int16_to_NumpyLong',
    NumpyInt32Type()      : 'Int32_to_NumpyLong',
    NumpyInt64Type()      : 'Int64_to_NumpyLong',
    NumpyFloat32Type()    : 'Float_to_NumpyDouble',
    NumpyFloat64Type()    : 'Double_to_NumpyDouble',
    NumpyComplex64Type()  : 'Complex64_to_NumpyComplex',
    NumpyComplex128Type() : 'Complex128_to_NumpyComplex'
    })

pytype_parse_registry.update({
    NumpyInt8Type()       : 'b',
    NumpyInt16Type()      : 'h',
    NumpyInt32Type()      : 'i',
    NumpyInt64Type()      : 'l',
    NumpyFloat32Type()    : 'f',
    NumpyFloat64Type()    : 'd',
    NumpyComplex64Type()  : 'O',
    NumpyComplex128Type() : 'O'
    })
