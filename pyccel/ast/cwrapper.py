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
    'PyErr_SetString',
#------- CAST FUNCTIONS ------
    'C_to_Python',
    'Python_to_C',
#-------CHECK FUNCTIONS ------
    'PythonType_Check',
#-------- Regestry -----------
    'flags_registry'
)


#-------------------------------------------------------------------
#                        Python DataTypes
#-------------------------------------------------------------------

class PyccelPyObject(DataType):
    _name = 'pyobject'

class PyccelPyArrayObject(DataType):
    """ Datatype representing a PyArrayObject which is the
    class used to hold numpy objects"""
    _name = 'pyarrayobject'


#-------------------------------------------------------------------
#                  Parsing and Building Classes
#-------------------------------------------------------------------

# TODO: Is there an equivalent to static so this can be a static list of strings?
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
        self._name      = name
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
    converter_functions : dict
        dictionary maping argument to cast functions
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

    converter_functions : dict
        dictionary maping argument to cast functions
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
#                      Python.h Constants
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True  = Variable(PyccelPyObject(), 'Py_True',is_pointer=True)
Py_False = Variable(PyccelPyObject(), 'Py_False',is_pointer=True)

# Python.h object representing None
Py_None  = Variable(PyccelPyObject(), 'Py_None', is_pointer=True)

#-------------------------------------------------------------------
#                      cwrapper.h functions
#-------------------------------------------------------------------

def Python_To_C(c_object):
    """
    Create FunctionDef responsible for casting python argument to C
    Parameters:
    ----------
    c_object  : Variable
        The variable needed for the generation of the cast_function
    Returns
    -------
    FunctionDef : cast type FunctionDef
    """
    try :
        cast_function = py_to_c_registry[(c_object.dtype, c_object.precision)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')
    cast_func = FunctionDef(name = check_type,
                       body      = [],
                       arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True),
                                    Variable(dtype=c_object.dtype, name = 'v',
                                            precision = c_object.precision, is_pointer=True)],
                       results   = [Variable(dtype=NativeBool(), name = 'r')])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
py_to_c_registry = {
    (NativeBool(), 4)    : 'PyBool_to_Bool',
    (NativeInteger(), 1) : 'PyInt8_to_Int8',
    (NativeInteger(), 2) : 'PyInt16_to_Int16',
    (NativeInteger(), 4) : 'PyInt32_to_Int32',
    (NativeInteger(), 8) : 'PyInt64_to_Int64',
    (NativeReal(), 4)    : 'PyFloat_to_Float',
    (NativeReal, 8)      : 'PyDouble_to_Double',
    (NativeComplex, 4)   : 'PyComplex_to_Complex64',
    (NativeComplex, 8)   : 'PyComplex_to_Complex128'}


def C_to_Python(c_object)
    """
    Create FunctionDef responsible for casting c argument to python
    Parameters:
    ----------
    c_object  : Variable
        The variable needed for the generation of the cast_function
    Returns
    -------
    FunctionDef : cast type FunctionDef
    """
    try :
        cast_function = c_to_py_registry[(c_object.dtype, c_object.precision)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')
    cast_func = FunctionDef(name = check_type,
                       body      = [],
                       arguments = [Variable(dtype=c_object.dtype, name = 'v', precision = c_object.precision)],
                       results   = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
# TODO create cast functions of different precision of int issue #735
c_to_py_registry = {
    (NativeBool(), 4)    : 'Bool_to_PyBool',
    (NativeInteger(), 1) : 'Int_to_pyLong,
    (NativeInteger(), 2) : 'Int_to_pyLong,
    (NativeInteger(), 4) : 'Int_to_PyLong',
    (NativeInteger(), 8) : 'Int_to_PyLong',
    (NativeReal(), 4)    : 'Double_to_PyDouble',
    (NativeReal, 8)      : 'Double_to_PyDouble',
    (NativeComplex, 4)   : 'Complex_to_PyComplex',
    (NativeComplex, 8)   : 'Complex_to_PyComplex'}


#-------------------------------------------------------------------
#              errors and check functions
#-------------------------------------------------------------------

def PythonType_Check(c_object, py_object):
    """
    Create FunctionCall responsible for checking python argument data type
    Parameters:
    ----------
    c_object  : Variable
        The variable needed for the generation of the type check
    py_object : Variable
        The python argument of the check function
    Returns
    -------
    FunctionCall : Check type FunctionCall
    """
    try :
        check_type = check_type_registry[c_object.dtype]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')
    check_func = FunctionDef(name = check_type,
                    body      = [],
                    arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                    results   = [Variable(dtype=NativeBool(), name = 'r')])

    return FunctionCall(check_func, [py_object])

# All functions used for type are from c python api :
# https://docs.python.org/3/c-api/long.html#c.PyLong_Check
# https://docs.python.org/3/c-api/complex.html#c.PyComplex_Check
# https://docs.python.org/3/c-api/float.html#c.PyFloat_Check
# https://docs.python.org/3/c-api/bool.html#c.PyBool_Check

check_type_registry  = {
    NativeInteger() : 'PyLong_Check',
    NativeComplex() : 'PyComplex_Check',
    NativeReal()    : 'PyFloat_Check',
    NativeBool()    : 'PyBool_Check',
}


def PyErr_SetString(exception, message):
    """
    Generate function Call of c/python api PyErr_SetString
    https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString
    used to set the error indicator.

    Parameters:
    ----------
    exception  : str
        The exception type
    message    : str
        Error message
    Returns
    FunctionCall : raise error FunctionCall
    """
    func = FunctionDef(name = 'PyErr_SetString',
                  body      = [],
                  arguments = [Variable(dtype = PyccelPyObject(), name = 'o'),
                               Variable(dtype = NativeString(), name = 's')],
                  results   = [])

    exception = Variable(PyccelPyObject(), name = exception)

    return FunctionCall(func, [exception, message])

# This registry is used for interface management,
# mapping each data type to a given flag
# Thos flag are used in a betset #TODO
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
