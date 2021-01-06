# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
from .basic import PyccelAstNode
from .datatypes import NativeInteger, default_precision

__all__ = (
    'PyccelInternalFunction',
    'PyccelArraySize'
)


class PyccelInternalFunction(PyccelAstNode):
    """ Abstract class used by function calls
    which are translated to Pyccel objects
    """
    def __init__(self, *args):
        self._args   = tuple(args)
        PyccelAstNode.__init__(self, *args)

    @property
    def args(self):
        return self._args


class PyccelArraySize(PyccelInternalFunction):

    def __init__(self, arg, index):
        if not isinstance(arg, (list,
                                tuple,
                                PyccelAstNode)):
            raise TypeError('Unknown type of  %s.' % type(arg))

        PyccelInternalFunction.__init__(self, arg, index)
        self._arg   = arg
        self._index = index
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        return self._arg

    @property
    def index(self):
        return self._index

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))
