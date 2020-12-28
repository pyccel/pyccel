#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
from sympy.core.function      import Function
from .basic                   import Basic, PyccelAstNode
from .datatypes import NativeInteger, default_precision

class PyccelArraySize(Function, PyccelAstNode):
    def __new__(cls, arg, index):
        is_PyccelAstNode = isinstance(arg, PyccelAstNode) and \
                (arg.shape is None or all(a.shape is None for a in arg.shape))
        if not (is_PyccelAstNode or hasattr(arg, '__len__')):
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg, index)

    def __init__(self, arg, index):
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))
