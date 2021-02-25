#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring

"""
Handling the transitions between python code and C code.
"""

import numpy as np

from ..errors.errors   import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

from .basic     import Basic

from .datatypes import DataType
from .datatypes import NativeInteger, NativeReal, NativeComplex
from .datatypes import NativeBool, NativeString, NativeGeneric, NativeVoid

from .core      import FunctionCall, FunctionDef, FunctionAddress

from .variable  import Variable, ValuedVariable

errors = Errors()

__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
    'PyccelPyArrayObject',
    'PyArgKeywords',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode',
#--------- CONSTANTS ----------
    'Py_True',
    'Py_False',
    'Py_None',
#----- C / PYTHON FUNCTIONS ---
    'pycomplex_real',
    'pycomplex_imag',
    'pycomplex_fromdoubles',
    'Py_DECREF',
    'PyLong_AsLong',
    'PyFloat_AsDouble',
    'PythonType_Check',
    'NumpyType_Check',
    'PyErr_SetString',
#------- CAST FUNCTIONS ------
    'pyint_to_bool',
    'bool_to_pyobj',
    'pycomplex_to_complex',
    'complex_to_pycomplex',
    'pybool_to_bool',
)


class PyccelPyObject(DataType):
    _name = 'pyobject'

class PyccelPyArrayObject(DataType):
    """ Datatype representing a PyArrayObject which is the
    class used to hold numpy objects"""
    _name = 'pyarrayobject'

#TODO: Is there an equivalent to static so this can be a static list of strings?
class PyArgKeywords(Basic):
    """
    Represents the list containing the names of all arguments to a function.
    This information allows the function to be called by keyword

    Parameters
    ----------
    name : str
        The name of the variable in which the list is stored
    arg_names : list of str
        A list of the names of the function arguments
    """
    _attribute_nodes = ()
    def __init__(self, name, arg_names):
        self._name = name
        self._arg_names = arg_names
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def arg_names(self):
        return self._arg_names

class PyArg_ParseTupleNode(Basic):
    """
    Represents a call to the function from Python.h which collects the expected arguments

    Parameters
    ----------
    python_func_args: Variable
        Args provided to the function in python
    python_func_kwargs: Variable
        Kwargs provided to the function in python
    parse_args: list of Variable
        List of arguments into which the result will be collected
    arg_names : list of str
        A list of the names of the function arguments
    """

    _attribute_nodes = ('_pyarg','_pykwarg','_parse_args','_arg_names')

    def __init__(self, python_func_args,
                       python_func_kwargs,
                       converter_functions,
                       parse_args,
                       arg_names):

        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not isinstance(converter_functions):
            raise TypeError('converter_functions should be a Dictionary')
        if any(not isinstance(f, FunctionDef) for f in converter_functions.keys()):
            raise TypeError('Converter function should be a FunctionDef')
        if not isinstance(parse_args, list) and any(not isinstance(c, Variable) for c in parse_args):
            raise TypeError('Parse args should be a list of Variables')
        if not isinstance(arg_names, PyArgKeywords):
            raise TypeError('Parse args should be a list of Variables')

        self._flags = ''
        i           = 0
        args_count  = len(parse_args)

        while i < args_count and not isinstance(parse_args[i], ValuedVariable):
            self._flags += 'O&'
            i += 1
        if i < args_count:
            self._flags += '|'
        while i < args_count:
            self._flags += 'O&' #TODO ? when iterface are back this should be in a function
            i += 1
        # Restriction as of python 3.8
        if any([isinstance(a, (Variable, FunctionAddress)) and a.is_kwonly for a in parse_args]):
            errors.report('Kwarg only arguments without default values will not raise an error if they are not passed',
                          symbol=parse_args, severity='warning')

        parse_args = [[converter_functions[get_custom_key(variable)] , a] for a in parse_args]
        parse_args = [a for arg in parse_args for a in arg]

        self._pyarg      = python_func_args
        self._pykwarg    = python_func_kwargs
        self._parse_args = parse_args
        self._arg_names  = arg_names
        super().__init__()

    @property
    def pyarg(self):
        return self._pyarg

    @property
    def pykwarg(self):
        return self._pykwarg

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._parse_args

    @property
    def arg_names(self):
        return self._arg_names


class PyBuildValueNode(Basic):
    """
    Represents a call to the function from Python.h which create a new value based on a format string

    Parameters
    ---------
    parse_args: list of Variable
        List of arguments which the result will be buit from
    """
    _attribute_nodes = ('_result_args',)

    def __init__(self, result_args = (), converter_functions = {}):
        self._flags       = ''
        for i in result_args:
            self._flags  += 'O&'

        result_args       = [[converter_functions[xxxxx], arg] for arg in result_args]
        self._result_args = [a for arg in result_args for a in arg]

        super().__init__()

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._result_args


def get_custom_key(variable):
    """
    """
    dtype     = variable.dtype
    precision = variable.precision
    rank      = variable.rank #TODO find a global way to manage different rank in one function
    is_valued = isinstance(variable, ValuedVariable)

    return (dtype, precision, rank, is_valued)


#-------------------------------------------------------------------
#                      Python.h functions
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True  = Variable(PyccelPyObject(), 'Py_True',is_pointer=True)
Py_False = Variable(PyccelPyObject(), 'Py_False',is_pointer=True)

# Python.h object representing None
Py_None  = Variable(PyccelPyObject(), 'Py_None', is_pointer=True)





# Casting functions
# Represents type of cast function responsible of the conversion of one data type into another.
# More information are in pyccel/stdlib/cwrapper/

c_bool_type        = Variable(dtype=NativeInteger(),  name = 'BOOL', precision = 4)
c_int8_type        = Variable(dtype=NativeInteger(),  name = 'INT8', precision = 4)
c_int16_type       = Variable(dtype=NativeInteger(),  name = 'INT16', precision = 4)
c_int32_type       = Variable(dtype=NativeInteger(),  name = 'INT32', precision = 4)
c_int64_type       = Variable(dtype=NativeInteger(),  name = 'INT64', precision = 4)
c_float_type       = Variable(dtype=NativeInteger(),  name = 'FLOAT', precision = 4)
c_double_type      = Variable(dtype=NativeInteger(),  name = 'DOUBLE', precision = 4)
c_cfloat_type      = Variable(dtype=NativeInteger(),  name = 'CFLOAT', precision = 4)
c_cdouble_type     = Variable(dtype=NativeInteger(),  name = 'CDOUBLE', precision = 4)

PyObject_AsCtype    = FunctionDef(name     = 'PyObject_AsCtype',
                                arguments  = [Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)
                                              Variable(dtype=NativeVoid(), name = 'v', precision = 4,
                                                       is_pointer = True),
                                              Variable(dtype=NativeInteger(), name = 'type')],
                                body       = [],
                                results    = [Variable(dtype=NativeBool(), name = 'b')])

PyObject_from_Ctype = FunctionDef(name     = 'PyObject_from_Ctype',
                                arguments  = [Variable(dtype=NativeBool(), name = 'b')],
                                body       = [],
                                results    = [Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)])

pyarray_to_ndarray  = FunctionDef(name     = 'PyArray_to_ndarray',
                                arguments  = [Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)],
                                body       = [],
                                results    = [])#TODO


cast_function_registry = {
    'pyint_to_bool' : pyint_to_bool,
    'bool_to_pyobj' : bool_to_pyobj,
    'pycomplex_to_complex' : pycomplex_to_complex,
    'complex_to_pycomplex': complex_to_pycomplex,
    'pybool_to_bool' : pybool_to_bool,
    'pyarray_to_ndarray' : pyarray_to_ndarray,
}

check_type_registry  = {
    NativeInteger() : 'PyLong_Check',
    NativeComplex() : 'PyComplex_Check',
    NativeReal()    : 'PyFloat_Check',
    NativeBool()    : 'PyBool_Check',
}

flags_registry = {
    (NativeInteger(), 4)       : 1,
    (NativeInteger(), 8)       : 2,
    (NativeInteger(), 2)       : 3,
    (NativeInteger(), 1)       : 4,
    (NativeReal(), 8)          : 5,
    (NativeReal(), 4)          : 6,
    (NativeComplex(), 4)       : 7,
    (NativeComplex(), 8)       : 8,
    (NativeBool(), 4)          : 9,
    (NativeString(), 0)        : 10
}
