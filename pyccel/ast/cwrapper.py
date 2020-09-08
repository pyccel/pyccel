from .basic     import Basic
from .datatypes import DataType
from .core import FunctionCall, FunctionDef
__all__ = (
#
# --------- CLASSES -----------
#
    'PyccelPyObject',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode'
)

class PyArg_ParseTupleNode(Basic):

    def __new__(cls, python_func_args, type_keys='', parse_args=None):
        return Basic.__new__(cls, python_func_args, type_keys, parse_args)        
 
    @property
    def pyarg(self):
        return self._args[0]

    @property
    def flags(self):
        return self._args[1]
    
    @property
    def args(self):
        return self._args[2]

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