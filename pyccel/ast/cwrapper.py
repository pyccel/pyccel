from .basic     import Basic
from .datatypes import DataType
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


class PyBuildValueNode():
    pass
class PyccelPyObject(DataType):
    _name = 'pyobject'