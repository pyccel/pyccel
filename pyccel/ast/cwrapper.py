from .basic     import Basic
from .datatypes import DataType
from .core      import FunctionCall, FunctionDef, Variable

__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode'
)

pytype_parse_registry = {
        NativeInteger(): 'l',
        NativeReal(): 'd',
        NativeComplex():'c',
        NativeBool():'p',
        NativeString():'s',
        PyccelPyObject():'O'
        }

class PyArg_ParseTupleNode(Basic):

    def __init__(self, python_func_args, c_func_args, parse_args, arg_names):
        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        self._pyarg = python_func_args
        if not isinstance(c_func_args, list) and any(not isinstance(c, Variable) for c in c_func_args):
            raise TypeError('C func args should be a list of Variables')
        if not isinstance(parse_args, list) and any(not isinstance(c, Variable) for c in parse_args):
            raise TypeError('Parse args should be a list of Variables')

        if len(parse_args) != len(c_func_args):
            raise TypeError('There should be the same number of c_func_args and parse_args')

        self._pyarg      = python_func_args
        self._parse_args = parse_args
        self._arg_names  = arg_names
        self._flags      = ''
        while i < len(c_func_args) and not c_func_args[i].is_kwonly:
            self._flags += pytype_parse_registry[c_func_args[i].dtype]
        if i < len(c_func_args):
            self._flags += '|'
        while i < len(c_func_args):
            self._flags += pytype_parse_registry[c_func_args[i].dtype]

    @property
    def pyarg(self):
        return self._pyarg

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._parse_args

#testing
class PyBuildValue(FunctionDef):
    def __new__(cls, flags, args, results = None, body = None):
        return FunctionDef.__new__(cls, name = 'Py_BuildValue', arguments = [flags] + args, results = results, body = body)



class PyBuildValueNode(Basic):
    def __new__(cls, build_keys='', res_args=None):
        return Basic.__new__(cls, build_keys, res_args)

    @property
    def flags(self):
        return self._args[0]

    @property
    def args(self):
        return self._args[1]


class PyccelPyObject(DataType):
    _name = 'pyobject'
