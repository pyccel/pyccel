# pylint: disable=missing-function-docstring, missing-module-docstring/

from .basic     import Basic

from pyccel.ast.numbers   import BooleanTrue, Complex
from .builtins  import PythonBool

from .datatypes import DataType
from .datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool, NativeString

from .core      import FunctionCall, FunctionDef, Variable, ValuedVariable, VariableAddress, FunctionAddress
from .core      import AliasAssign, Assign, Return
from .core      import PyccelEq, If

from .numpyext  import NumpyReal, NumpyImag

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *


errors = Errors()

__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
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
    'PyType_Check',
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
    def __init__(self, name, arg_names):
        self._name = name
        self._arg_names = arg_names

    @property
    def name(self):
        return self._name

    @property
    def arg_names(self):
        return self._arg_names

#using the documentation of PyArg_ParseTuple() and Py_BuildValue https://docs.python.org/3/c-api/arg.html
pytype_parse_registry = {
    (NativeInteger(), 4) : 'i',
    (NativeInteger(), 8) : 'l',
    (NativeInteger(), 2) : 'h',
    (NativeInteger(), 1) : 'b',
    (NativeReal(), 8)    : 'd',
    (NativeReal(), 4)    : 'f',
    (NativeComplex(), 4) : 'O',
    (NativeComplex(), 8) : 'O',
    (NativeBool(), 4)    : 'p',
    (NativeString(), 0)  : 's',
    (PyccelPyObject(), 0): 'O',
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
    """

    def __init__(self, python_func_args, python_func_kwargs, c_func_args, parse_args, arg_names):
        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not all(isinstance(c, (Variable, FunctionAddress)) for c in c_func_args):
            raise TypeError('C func args should be a list of Variables')
        if not isinstance(parse_args, list) and any(not isinstance(c, Variable) for c in parse_args):
            raise TypeError('Parse args should be a list of Variables')
        if not isinstance(arg_names, PyArgKeywords):
            raise TypeError('Parse args should be a list of Variables')

        if len(parse_args) != len(c_func_args):
            raise TypeError('There should be the same number of c_func_args and parse_args')

        self._pyarg      = python_func_args
        self._pykwarg    = python_func_kwargs
        self._parse_args = parse_args
        self._arg_names  = arg_names
        self._flags      = ''
        i = 0

        while i < len(c_func_args) and not isinstance(c_func_args[i], ValuedVariable):
            if isinstance(c_func_args[i], FunctionAddress):
                self._flags += 'O'
            else:
                self._flags += pytype_parse_registry[(parse_args[i].dtype, parse_args[i].precision)]
            i+=1
        if i < len(c_func_args):
            self._flags += '|'
        while i < len(c_func_args):
            if isinstance(c_func_args[i], FunctionAddress):
                self._flags += 'O'
            else:
                self._flags += pytype_parse_registry[(parse_args[i].dtype, parse_args[i].precision)]
            i+=1

        # Restriction as of python 3.8
        if any([isinstance(a, (Variable, FunctionAddress)) and a.is_kwonly for a in c_func_args]):
            errors.report('Kwarg only arguments without default values will not raise an error if they are not passed',
                          symbol=c_func_args, severity='warning')

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

    def __init__(self, result_args = []):
        self._flags = ''
        self._result_args = result_args
        for i in result_args:
            self._flags += pytype_parse_registry[(i.dtype, i.precision)]

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._result_args

#-------------------------------------------------------------------
#                      Python.h functions
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True = Variable(PyccelPyObject(), 'Py_True',is_pointer=True)
Py_False = Variable(PyccelPyObject(), 'Py_False',is_pointer=True)

# Python.h object representing None
Py_None = Variable(PyccelPyObject(), 'Py_None', is_pointer=True)

# Python.h function managing complex data type
pycomplex_real = FunctionDef(name      = 'PyComplex_RealAsDouble',
                           body      = [],
                           arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                           results   = [Variable(dtype=NativeReal(), name = 'r')])
pycomplex_imag = FunctionDef(name      = 'PyComplex_ImagAsDouble',
                           body      = [],
                           arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                           results   = [Variable(dtype=NativeReal(), name = 'r')])
pycomplex_fromdoubles = FunctionDef(name      = 'PyComplex_FromDoubles',
                           body      = [],
                           arguments = [Variable(dtype=NativeReal(), name = 'r'),
                                        Variable(dtype=NativeReal(), name = 'i')],
                           results   = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)])
Py_DECREF = FunctionDef(name = 'Py_DECREF',
                        body = [],
                        arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                        results = [])
PyLong_AsLong = FunctionDef(name = 'PyLong_AsLong',
                        body = [],
                        arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                        results   = [Variable(dtype=NativeInteger(), name = 'r')])
PyFloat_AsDouble = FunctionDef(name = 'PyFloat_AsDouble',
                        body = [],
                        arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                        results   = [Variable(dtype=NativeReal(), name = 'r')])

def PyType_Check(data_type):
    try :
        check_type = check_type_registry[data_type]
    except KeyError:
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=data_type,severity='fatal')
    func = FunctionDef(name = check_type,
                    body = [],
                    arguments = [Variable(dtype=PyccelPyObject(), name = 'o', is_pointer=True)],
                    results   = [Variable(dtype=NativeBool(), name = 'r')])
    return func

def PyErr_SetString(error_type, error_msg):
    func = FunctionDef(name = 'PyErr_SetString',
                body = [],
                arguments = [Variable(dtype=PyccelPyObject(), name = 'o'),
                             Variable(dtype =NativeString(), name = 's')],
                results   = [])
    err_type = Variable(PyccelPyObject(), error_type)
    return FunctionCall(func, [err_type, error_msg])

# Casting functions
# Represents type of cast function responsible of the conversion of one data type into another.
# Parameters :
# name of function , list of arguments ,  list of results

def pyint_to_bool(cast_function_name):
    cast_function_argument = Variable(dtype=NativeInteger(), name = 'i', precision=4)
    cast_function_result   = Variable(dtype=NativeBool(), name = 'b')
    cast_function_body     = [Assign(cast_function_result, PythonBool(cast_function_argument)),
                              Return([cast_function_result])  ]
    return FunctionDef(name      = cast_function_name,
                       arguments = [cast_function_argument],
                       body      = cast_function_body,
                       results   = [cast_function_result])

def bool_to_pyobj(cast_function_name):
    cast_function_argument = Variable(dtype=NativeBool(), name = 'b')
    cast_function_result   = Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)
    cast_function_body = [If(
                            (PythonBool(cast_function_argument),
                                [AliasAssign(cast_function_result, Py_True)]),
                            (BooleanTrue(),
                                [AliasAssign(cast_function_result, Py_False)])
                          ),
                          Return([cast_function_result])]
    return FunctionDef(name      = cast_function_name,
                       arguments = [cast_function_argument],
                       body      = cast_function_body,
                       results   = [cast_function_result])

def complex_to_pycomplex(cast_function_name):
    cast_function_argument = Variable(dtype=NativeComplex(), name = 'c')
    cast_function_result   = Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)
    real_part = Variable(dtype = NativeReal(), name = 'real_part')
    imag_part = Variable(dtype = NativeReal(), name = 'imag_part')
    cast_function_local_vars = [real_part, imag_part]

    cast_function_body = [Assign(real_part, NumpyReal(cast_function_argument)),
                          Assign(imag_part, NumpyImag(cast_function_argument)),
                          AliasAssign(cast_function_result,
                              FunctionCall(pycomplex_fromdoubles, [real_part, imag_part])),
                          Return([cast_function_result])]
    return FunctionDef(name       = cast_function_name,
                       arguments  = [cast_function_argument],
                       body       = cast_function_body,
                       results    = [cast_function_result],
                       local_vars = cast_function_local_vars)

def pycomplex_to_complex(cast_function_name):
    cast_function_argument = Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)
    cast_function_result   = Variable(dtype=NativeComplex(), name = 'c')
    real_part = Variable(dtype = NativeReal(), name = 'real_part')
    imag_part = Variable(dtype = NativeReal(), name = 'imag_part')
    cast_function_local_vars = [real_part, imag_part]

    cast_function_body = [Assign(real_part, FunctionCall(pycomplex_real, [cast_function_argument])),
                          Assign(imag_part, FunctionCall(pycomplex_imag, [cast_function_argument])),
                          Assign(cast_function_result, Complex(real_part, imag_part)),
                          Return([cast_function_result])]
    return FunctionDef(name      = cast_function_name,
                       arguments = [cast_function_argument],
                       body      = cast_function_body,
                       results   = [cast_function_result],
                       local_vars= cast_function_local_vars)

def pybool_to_bool(cast_function_name):
    cast_function_argument = Variable(dtype=PyccelPyObject(), name='o', is_pointer=True)
    cast_function_result   = Variable(dtype=NativeBool(), name = 'c')

    cast_function_body = [Assign(cast_function_result , PyccelEq(VariableAddress(cast_function_argument), VariableAddress(Py_True)))
                        , Return([cast_function_result])]

    return FunctionDef(name      = cast_function_name,
                       arguments = [cast_function_argument],
                       body      = cast_function_body,
                       results   = [cast_function_result])

cast_function_registry = {
    'pyint_to_bool' : pyint_to_bool,
    'bool_to_pyobj' : bool_to_pyobj,
    'pycomplex_to_complex' : pycomplex_to_complex,
    'complex_to_pycomplex': complex_to_pycomplex,
    'pybool_to_bool' : pybool_to_bool,
}

collect_function_registry = {
    NativeInteger(): PyLong_AsLong,
    NativeReal() : PyFloat_AsDouble,
}

check_type_registry = {
    NativeInteger(): 'PyLong_Check',
    NativeComplex() : 'PyComplex_Check',
    NativeReal() : 'PyFloat_Check',
    NativeBool() : 'PyBool_Check',
}
