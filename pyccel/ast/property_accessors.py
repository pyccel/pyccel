# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module containing commonly used objects which provide access to properties
of other objects.
"""
from sympy.core.function    import Function
from .basic                 import Basic, PyccelAstNode
from .datatypes             import NativeInteger, default_precision

__all__ = (
        'PyccelArraySize',
        'Slice'
)

class PyccelArraySize(Function, PyccelAstNode):
    """
    Class representing a call to a function which would
    return the shape of an object in a given dimension

    Parameters
    ==========
    arg   : PyccelAstNode
            A PyccelAstNode of unknown shape
    index : int
            The dimension along which the shape is
            provided
    """
    def __new__(cls, arg, index):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        if index >= arg.rank:
            raise TypeError('Index {} out of bounds for object {}'.format(index,arg))

        return Basic.__new__(cls, arg, index)

    def __init__(self, arg, index):
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        """ Object whose shape is calculated
        """
        return self._args[0]

    @property
    def index(self):
        """ Dimension in which the shape is calculated
        """
        return self._args[1]

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))


class Slice(Basic, PyccelAstNode):

    """Represents a slice in the code.

    Parameters
    ----------
    start : Symbol or int
        starting index

    stop : Symbol or int
        ending index

    step : Symbol or int default None

    Examples
    --------
    >>> from sympy import symbols
    >>> from pyccel.ast.core import Slice
    >>> start, end, step = symbols('start, stop, step', integer=True)
    >>> Slice(start, stop)
    start : stop
    >>> Slice(None, stop)
     : stop
    >>> Slice(start, None)
    start :
    >>> Slice(start, stop, step)
    start : stop : step
    """

    def __new__(cls, start, stop, step = None):
        return Basic.__new__(cls, start, stop, step)

    def __init__(self, start, stop, step = None):
        self._start = start
        self._stop = stop
        self._step = step
        if self.stage == 'syntactic':
            return
        if start is not None and not (hasattr(start, 'dtype') and isinstance(start.dtype, NativeInteger)):
            raise TypeError('Slice start must be Integer or None')
        if stop is not None and not (hasattr(stop, 'dtype') and isinstance(stop.dtype, NativeInteger)):
            raise TypeError('Slice stop must be Integer or None')
        if step is not None and not (hasattr(step, 'dtype') and isinstance(step.dtype, NativeInteger)):
            raise TypeError('Slice step must be Integer or None')

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.start is None:
            start = ''
        else:
            start = sstr(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = sstr(self.stop)
        return '{0} : {1}'.format(start, stop)

    def __str__(self):
        if self.start is None:
            start = ''
        else:
            start = str(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = str(self.stop)
        return '{0} : {1}'.format(start, stop)
