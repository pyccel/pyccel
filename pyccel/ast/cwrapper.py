#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module representing objects (functions/variables etc) required for the interface
between Python code and C code (using Python/C Api and cwrapper.c).
"""

from ..errors.errors import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

from .basic     import Basic, PyccelAstNode

from .datatypes import DataType, default_precision
from .datatypes import NativeInteger, NativeFloat, NativeComplex
from .datatypes import NativeBool, NativeString

from .core      import FunctionDefArgument, FunctionDefResult
from .core      import FunctionCall, FunctionDef, FunctionAddress
from .core      import Module

from .internals import get_final_precision

from .variable  import Variable

from .c_concepts import ObjectAddress


errors = Errors()

__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
    'PyArgKeywords',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode',
    'PyModule_AddObject',
#--------- CONSTANTS ----------
    'Py_True',
    'Py_False',
    'Py_None',
    'flags_registry',
#----- C / PYTHON FUNCTIONS ---
    'Py_DECREF',
    'set_python_error_message',
#----- CHECK FUNCTIONS ---
    'generate_datatype_error',
    'scalar_object_check',
)

#-------------------------------------------------------------------
#                        Python DataTypes
#-------------------------------------------------------------------
class PyccelPyObject(DataType):
    """ Datatype representing a PyObject which is the
    class used to hold python objects"""
    __slots__ = ()
    _name = 'pyobject'

#-------------------------------------------------------------------
#                  Parsing and Building Classes
#-------------------------------------------------------------------

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
    __slots__ = ('_name','_arg_names')
    _attribute_nodes = ()
    def __init__(self, name, arg_names):
        self._name = name
        self._arg_names = arg_names
        super().__init__()

    @property
    def name(self):
        """ The name of the variable in which the list of
        all arguments to the function is stored
        """
        return self._name

    @property
    def arg_names(self):
        """ The names of the arguments to the function which are
        contained in the PyArgKeywords list
        """
        return self._arg_names

#using the documentation of PyArg_ParseTuple() and Py_BuildValue https://docs.python.org/3/c-api/arg.html
pytype_parse_registry = {
    (NativeInteger(), 4)       : 'i',
    (NativeInteger(), 8)       : 'l',
    (NativeInteger(), 2)       : 'h',
    (NativeInteger(), 1)       : 'b',
    (NativeFloat(), 8)         : 'd',
    (NativeFloat(), 4)         : 'f',
    (NativeComplex(), 4)       : 'O',
    (NativeComplex(), 8)       : 'O',
    (NativeBool(), 4)          : 'p',
    (NativeString(), 0)        : 's',
    (PyccelPyObject(), 0)      : 'O',
    }

class PyArg_ParseTupleNode(Basic):
    """
    Represents a call to the function from Python.h which collects the expected arguments

    Parameters
    ----------
    python_func_args: Variable
        Args provided to the function in python
    python_func_kwargs: Variable
        Kwargs provided to the function in python
    c_func_args: list of Variable
        List of expected arguments. This helps determine the expected output types
    parse_args: list of Variable
        List of arguments into which the result will be collected
    arg_names : list of str
        A list of the names of the function arguments
    is_interface : boolean
        Default value False and True when working with interface functions
    """
    __slots__ = ('_pyarg','_pykwarg','_parse_args','_arg_names','_flags')
    _attribute_nodes = ('_pyarg','_pykwarg','_parse_args','_arg_names')

    def __init__(self, python_func_args,
                        python_func_kwargs,
                        c_func_args,
                        parse_args,
                        arg_names):
        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not isinstance(parse_args, list) and any(not isinstance(c, Variable) for c in parse_args):
            raise TypeError('Parse args should be a list of Variables')
        if not isinstance(arg_names, PyArgKeywords):
            raise TypeError('Parse args should be a list of Variables')

        i = 0
        while i < len(c_func_args) and not c_func_args[i].has_default:
            i+=1
        self._flags = 'O'*i

        if i < len(c_func_args):
            self._flags += '|'
            self._flags += 'O'*(len(c_func_args)-i)

        self._pyarg      = python_func_args
        self._pykwarg    = python_func_kwargs
        self._parse_args = parse_args
        self._arg_names  = arg_names
        super().__init__()

    def get_pytype(self, c_arg, parse_arg):
        """Return the needed flag to parse or build value
        """
        if isinstance(c_arg, FunctionAddress):
            return 'O'
        else:
            try:
                return pytype_parse_registry[(parse_arg.dtype, get_final_precision(parse_arg))]
            except KeyError as e:
                raise NotImplementedError("Type not implemented for argument collection : "+str(type(parse_arg))) from e

    @property
    def pyarg(self):
        """ The  variable containing all positional arguments
        passed to the function
        """
        return self._pyarg

    @property
    def pykwarg(self):
        """ The  variable containing all keyword arguments
        passed to the function
        """
        return self._pykwarg

    @property
    def flags(self):
        """ The flags indicating the types of the objects to
        be collected from the python arguments passed to the
        function
        """
        return self._flags

    @property
    def args(self):
        """ The arguments into which the python args and kwargs
        are collected
        """
        return self._parse_args

    @property
    def arg_names(self):
        """ The PyArgKeywords object which contains all the
        names of the function's arguments
        """
        return self._arg_names

class PyBuildValueNode(PyccelAstNode):
    """
    Represents a call to the function PyBuildValueNode.

    The function PyBuildValueNode can be found in Python.h.
    It describes the creation of a new Python object based
    on a format string. More details can be found in Python's
    docs.

    Parameters
    ----------
    result_args : list of Variable
        List of arguments which the result will be built from.
    """
    __slots__ = ('_flags','_result_args')
    _attribute_nodes = ('_result_args',)
    _dtype = PyccelPyObject
    _rank = 0
    _precision = 0
    _shape = ()
    _order = None

    def __init__(self, result_args = ()):
        self._flags = ''
        self._result_args = result_args
        for i in result_args:
            self._flags += pytype_parse_registry[(i.dtype, get_final_precision(i))]
        super().__init__()

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._result_args

#-------------------------------------------------------------------
class PyModule_AddObject(PyccelAstNode):
    """
    Represents a call to the PyModule_AddObject function.

    The PyModule_AddObject function can be found in Python.h.
    It adds a PythonObject to a module. More information about
    this function can be found in Python's documentation.

    Parameters
    ----------
    mod_name : str
                The name of the variable containing the module.
    name : str
                The name of the variable being added to the module.
    variable : Variable
                The variable containing the PythonObject.
    """
    __slots__ = ('_mod_name','_name','_var')
    _attribute_nodes = ('_name','_var')
    _dtype = NativeInteger()
    _precision = 4
    _rank = 0
    _shape = None

    def __init__(self, mod_name, name, variable):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if not isinstance(variable, Variable) or \
                variable.dtype is not PyccelPyObject():
            raise TypeError("Variable must be a PyObject Variable")
        self._mod_name = mod_name
        self._name = name
        self._var = ObjectAddress(variable)
        super().__init__()

    @property
    def mod_name(self):
        """ The name of the variable containing the module
        """
        return self._mod_name

    @property
    def name(self):
        """ The name of the variable being added to the module
        """
        return self._name

    @property
    def variable(self):
        """ The variable containing the PythonObject
        """
        return self._var

#-------------------------------------------------------------------
#                      Python.h Constants
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True = Variable(PyccelPyObject(), 'Py_True', memory_handling='alias')
Py_False = Variable(PyccelPyObject(), 'Py_False', memory_handling='alias')

# Python.h object representing None
Py_None = Variable(PyccelPyObject(), 'Py_None', memory_handling='alias')

# https://docs.python.org/3/c-api/refcounting.html#c.Py_DECREF
Py_DECREF = FunctionDef(name = 'Py_DECREF',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))],
                        results = [])

#-------------------------------------------------------------------
#                      cwrapper.h functions
#-------------------------------------------------------------------

def Python_to_C(c_object):
    """
    Create a FunctionDef responsible for casting scalar Python argument to C.

    Creates a FunctionDef node which contains all the code necessary
    for casting a PythonObject to a C object whose characteristics
    match that of the object passed as an argument.

    Parameters
    ----------
    c_object : Variable
        The variable needed for the generation of the cast_function.

    Returns
    -------
    FunctionDef
        The function which casts the Python object to C.
    """
    dtype = c_object.dtype
    prec  = get_final_precision(c_object)
    try :
        cast_function = py_to_c_registry[(dtype, prec)]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=dtype,severity='fatal')
    cast_func = FunctionDef(name = cast_function,
                       body      = [],
                       arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias'))],
                       results   = [FunctionDefResult(Variable(dtype=dtype, name = 'v', precision = prec))])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
py_to_c_registry = {
    (NativeBool(), 4)      : 'PyBool_to_Bool',
    (NativeInteger(), 1)   : 'PyInt8_to_Int8',
    (NativeInteger(), 2)   : 'PyInt16_to_Int16',
    (NativeInteger(), 4)   : 'PyInt32_to_Int32',
    (NativeInteger(), 8)   : 'PyInt64_to_Int64',
    (NativeFloat(), 4)     : 'PyFloat_to_Float',
    (NativeFloat(), 8)     : 'PyDouble_to_Double',
    (NativeComplex(), 4)   : 'PyComplex_to_Complex64',
    (NativeComplex(), 8)   : 'PyComplex_to_Complex128'}

def C_to_Python(c_object):
    """
    Create a FunctionDef responsible for casting scalar C results to Python.

    Creates a FunctionDef node which contains all the code necessary
    for casting a C object, whose characteristics match that of the object
    passed as an argument, to a PythonObject which can be used in Python code.

    Parameters
    ----------
    c_object : Variable
        The variable needed for the generation of the cast_function.

    Returns
    -------
    FunctionDef
        The function which casts the C object to Python.
    """
    if c_object.rank != 0:
        if c_object.order == 'C':
            cast_function = 'c_ndarray_to_pyarray'
        elif c_object.order == 'F':
            cast_function = 'fortran_ndarray_to_pyarray'
        else:
            cast_function = 'ndarray_to_pyarray'
    else:
        try :
            cast_function = c_to_py_registry[(c_object.dtype, c_object.precision)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')

    cast_func = FunctionDef(name = cast_function,
                       body      = [],
                       arguments = [FunctionDefArgument(c_object.clone('v', is_argument = True))],
                       results   = [FunctionDefResult(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias'))])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
c_to_py_registry = {
    (NativeBool(), -1)     : 'Bool_to_PyBool',
    (NativeBool(), 4)      : 'Bool_to_PyBool',
    (NativeInteger(), -1)  : 'Int'+str(default_precision['int']*8)+'_to_PyLong',
    (NativeInteger(), 1)   : 'Int8_to_NumpyLong',
    (NativeInteger(), 2)   : 'Int16_to_NumpyLong',
    (NativeInteger(), 4)   : 'Int32_to_NumpyLong',
    (NativeInteger(), 8)   : 'Int64_to_NumpyLong',
    (NativeFloat(), 4)     : 'Float_to_NumpyDouble',
    (NativeFloat(), 8)     : 'Double_to_NumpyDouble',
    (NativeFloat(), -1)    : 'Double_to_PyDouble',
    (NativeComplex(), 4)   : 'Complex64_to_NumpyComplex',
    (NativeComplex(), 8)   : 'Complex128_to_NumpyComplex',
    (NativeComplex(), -1)  : 'Complex128_to_PyComplex'}


#-------------------------------------------------------------------
#              errors and check functions
#-------------------------------------------------------------------

# https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred
PyErr_Occurred = FunctionDef(name      = 'PyErr_Occurred',
                             arguments = [],
                             results   = [FunctionDefResult(Variable(dtype = PyccelPyObject(), name = 'r', memory_handling = 'alias'))],
                             body      = [])

PyErr_SetString = FunctionDef(name = 'PyErr_SetString',
              body      = [],
              arguments = [FunctionDefArgument(Variable(dtype = PyccelPyObject(), name = 'o')),
                           FunctionDefArgument(Variable(dtype = NativeString(), name = 's'))],
              results   = [])

PyNotImplementedError = Variable(PyccelPyObject(), name = 'PyExc_NotImplementedError')
PyTypeError = Variable(PyccelPyObject(), name = 'PyExc_TypeError')

def set_python_error_message(exception, message):
    """
    Generate a function call which sets the Python error.

    Generate a function call of C/Python API PyErr_SetString
    https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString
    with a defined error message used to set the error indicator.

    Parameters
    ----------
    exception : str
        The error exception type.
    message : str
        The message which will be shown.

    Returns
    -------
    FunctionCall
        The FunctionCall which raises the error.
    """

    exception = Variable(PyccelPyObject(), name = exception)

    return FunctionCall(PyErr_SetString, [exception, message])


def generate_datatype_error(variable):
    """
    Generate TypeError exception from the variable information.

    Generate a TypeError exception indicated that the variable passed
    as an argument does not have the right datatype/precision.

    Parameters
    ----------
    variable : Variable
        The variable which indicates the correct datatype/precision.

    Returns
    -------
    FunctionCall
        Call to PyErr_SetString with TypeError as exception and custom message.
    """
    dtype     = variable.dtype

    if variable.precision == -1:
        precision = 'native '
    elif isinstance(dtype, NativeBool):
        precision = ''
    elif isinstance(dtype, NativeComplex):
        precision = '{} bit '.format(variable.precision * 2 * 8)
    else:
        precision = '{} bit '.format(variable.precision * 8)

    message = '"Argument must be {precision}{dtype}"'.format(
            precision = precision,
            dtype     = variable.dtype)
    return set_python_error_message('PyExc_TypeError', message)

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
py_to_c_registry = {
    (NativeBool(), 4)      : 'PyBool_to_Bool',
    (NativeInteger(), 1)   : 'PyInt8_to_Int8',
    (NativeInteger(), 2)   : 'PyInt16_to_Int16',
    (NativeInteger(), 4)   : 'PyInt32_to_Int32',
    (NativeInteger(), 8)   : 'PyInt64_to_Int64',
    (NativeFloat(), 4)     : 'PyFloat_to_Float',
    (NativeFloat(), 8)     : 'PyDouble_to_Double',
    (NativeComplex(), 4)   : 'PyComplex_to_Complex64',
    (NativeComplex(), 8)   : 'PyComplex_to_Complex128'}

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
check_type_registry = {
    (NativeBool(), -1)     : 'PyIs_Bool',
    (NativeBool(), 4)      : 'PyIs_Bool',
    (NativeInteger(), -1)  : 'PyIs_NativeInt',
    (NativeInteger(), 1)   : 'PyIs_Int8',
    (NativeInteger(), 2)   : 'PyIs_Int16',
    (NativeInteger(), 4)   : 'PyIs_Int32',
    (NativeInteger(), 8)   : 'PyIs_Int64',
    (NativeFloat(), -1)    : 'PyIs_NativeFloat',
    (NativeFloat(), 4)     : 'PyIs_Float',
    (NativeFloat(), 8)     : 'PyIs_Double',
    (NativeComplex(), -1)  : 'PyIs_NativeComplex',
    (NativeComplex(), 4)   : 'PyIs_Complex64',
    (NativeComplex(), 8)   : 'PyIs_Complex128'}

def scalar_object_check(py_object, c_object):
    """
    Create FunctionCall responsible for checking Python argument data type.

    Create a FunctionCall which checks whether the Python argument
    passed as an argument is a scalar with a type which matches the
    type of the C object.

    Parameters
    ----------
    py_object : Variable
        The Python argument of the check function.
    c_object : Variable
        The variable needed for the generation of the type check.

    Returns
    -------
    FunctionCall
        Check type FunctionCall.
    """

    try :
        check_type = check_type_registry[c_object.dtype, c_object.precision]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')

    check_func = FunctionDef(name = check_type,
                    body      = [],
                    arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling = 'alias'))],
                    results   = [FunctionDefResult(Variable(dtype=NativeBool(), name = 'r'))])

    return FunctionCall(check_func, [py_object])

class PyModule(Module):
    """
    Class to hold a module which is accessible from Python
    """
    __slots__ = ('_func_names', '_external_funcs')
    def __init__(self, *args, func_names, external_funcs, **kwargs):
        self._func_names = func_names
        self._external_funcs = external_funcs
        super().__init__(*args, **kwargs)

    @property
    def external_funcs(self):
        return self._external_funcs

class PyFunctionDef(FunctionDef):
    """
    Class to hold a FunctionDef which is accessible from Python

    Contains the Python-compatible version of the function which is
    used for the wrapper.
    As compared to a normal FunctionDef, this version contains
    arguments for the shape of arrays. It should be generated by
    calling `codegen.wrapper.CToPythonWrapper.wrap`.

    Parameters
    ----------
    *args : list
        See FunctionDef.

    original_function : FunctionDef
        The function from which the Python-compatible version was created.

    **kwargs : dict
        See FunctionDef.

    See Also
    --------
    pyccel.ast.core.FunctionDef
        The class from which BindCFunctionDef inherits which contains all
        details about the args and kwargs.
    """
    __slots__ = ('_original_function',)
    _attribute_nodes = (*FunctionDef._attribute_nodes, '_original_function')

    def __init__(self, *args, original_function, **kwargs):
        self._original_function = original_function
        super().__init__(*args, **kwargs)

    @property
    def original_function(self):
        """
        The function which is wrapped by this BindCFunctionDef.

        The original function which would be printed in pure Fortran which is not
        compatible with C.
        """
        return self._original_function
