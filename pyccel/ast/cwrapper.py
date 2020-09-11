from .basic     import Basic
from .datatypes import DataType
from .datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool, NativeString
from .core      import FunctionCall, FunctionDef, Variable, ValuedVariable

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
    'PyccelPyComplex',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode'
)

class PyccelPyObject(DataType):
    _name = 'pyobject'

class PyccelPyComplex(DataType):
    _name = 'pycomplex'

#TODO: Is there an equivalent to static so this can be a static list of strings?
class PyArgKeywords(Basic):
    def __init__(self, name, arg_names):
        self._name = name
        self._arg_names = arg_names

    @property
    def name(self):
        return self._name

    @property
    def arg_names(self):
        return self._arg_names

pytype_parse_registry = {
    (NativeInteger(), 4) : 'i',
    (NativeInteger(), 8) : 'l',
    (NativeInteger(), 2) : 'h',
    (NativeInteger(), 1) : 'b',
    (NativeReal(), 8)    : 'd',
    (NativeReal(), 4)    : 'f',
    (NativeComplex(), 4) : 'D',
    (NativeComplex(), 8) : 'D',
    (NativeBool(), 4)    : 'p',
    (NativeString(), 0)  : 's',
    (PyccelPyObject(), 0): 'O'
    }

class PyArg_ParseTupleNode(Basic):

    def __init__(self, python_func_args, python_func_kwargs, c_func_args, parse_args, arg_names):
        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not isinstance(c_func_args, list) and any(not isinstance(c, Variable) for c in c_func_args):
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
            self._flags += pytype_parse_registry[(c_func_args[i].dtype, c_func_args[i].precision)]
            i+=1
        if i < len(c_func_args):
            self._flags += '|'
        while i < len(c_func_args):
            self._flags += pytype_parse_registry[(c_func_args[i].dtype, c_func_args[i].precision)]
            i+=1

        # Restriction as of python 3.8
        if any([isinstance(a, Variable) and a.is_kwonly for a in c_func_args]):
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
        