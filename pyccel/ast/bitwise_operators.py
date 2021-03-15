#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module handling all python builtin operators
These operators all have a precision as detailed here:
    https://docs.python.org/3/reference/expressions.html#operator-precedence
They also have specific rules to determine the dtype, precision, rank, shape
"""
# TODO [EB 12.03.21]: Remove pylint command with PR #797
# pylint: disable=W0201
from .builtins     import PythonInt
from .datatypes    import (NativeBool, NativeInteger, NativeReal,
                           NativeComplex, NativeString)
from .operators     import PyccelUnaryOperator, PyccelOperator

__all__ = (
    'PyccelRShift',
    'PyccelLShift',
    'PyccelBitXor',
    'PyccelBitOr',
    'PyccelBitAnd',
    'PyccelInvert',
)

#==============================================================================

class PyccelInvert(PyccelUnaryOperator):
    """
    Class representing a call to the python bitwise not operator.
    I.e:
        ~a
    is equivalent to:
        PyccelInvert(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    __slots__ = ()
    _precedence = 14

    def _set_dtype(self):
        self._dtype     = NativeInteger()
        a = self._args[0]
        if a.dtype not in (NativeInteger(), NativeBool()):
            raise TypeError('unsupported operand type(s): {}'.format(self))

        self._args      = (PythonInt(a) if a.dtype is NativeBool() else a,)

        self._precision = a.precision

    def __repr__(self):
        return '~{}'.format(repr(self.args[0]))

#==============================================================================

class PyccelBitOperator(PyccelOperator):
    """ Abstract superclass representing a python
    bitwise operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()

    def _set_dtype(self):
        """ Sets the dtype and precision

        If one argument is a string then all arguments must be strings

        If the arguments are numeric then the dtype and precision
        match the broadest type and the largest precision
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex
        """
        integers  = [a for a in self._args if a.dtype in (NativeInteger(),NativeBool())]
        reals     = [a for a in self._args if a.dtype is NativeReal()]
        complexes = [a for a in self._args if a.dtype is NativeComplex()]
        strs      = [a for a in self._args if a.dtype is NativeString()]

        if strs or complexes or reals:
            raise TypeError('unsupported operand type(s): {}'.format(self))
        elif integers:
            self._handle_integer_type(integers)
        else:
            raise TypeError('cannot determine the type of {}'.format(self))

    def _set_shape_rank(self):
        self._rank = 0
        self._shape = ()

    def _handle_integer_type(self, integers):
        self._dtype     = NativeInteger()
        self._precision = max(a.precision for a in integers)
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in integers]

#==============================================================================

class PyccelRShift(PyccelBitOperator):
    """
    Class representing a call to the python right shift operator.
    I.e:
        a >> b
    is equivalent to:
        PyccelRShift(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 11

    def __repr__(self):
        return '{} >> {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelLShift(PyccelBitOperator):
    """
    Class representing a call to the python right shift operator.
    I.e:
        a << b
    is equivalent to:
        PyccelRShift(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 11

    def __repr__(self):
        return '{} << {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelBitComparisonOperator(PyccelBitOperator):
    """ Abstract superclass representing a python
    bitwise comparison operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    def _handle_integer_type(self, integers):
        if all(a.dtype is NativeInteger() for a in integers):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in integers):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in integers]
        self._precision = max(a.precision for a in integers)

#==============================================================================

class PyccelBitXor(PyccelBitComparisonOperator):
    """
    Class representing a call to the python bitwise XOR operator.
    I.e:
        a ^ b
    is equivalent to:
        PyccelBitXor(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 9

    def __repr__(self):
        return '{} ^ {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelBitOr(PyccelBitComparisonOperator):
    """
    Class representing a call to the python bitwise OR operator.
    I.e:
        a | b
    is equivalent to:
        PyccelBitOr(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 8

    def __repr__(self):
        return '{} | {}'.format(self.args[0], self.args[1])

#==============================================================================

class PyccelBitAnd(PyccelBitComparisonOperator):
    """
    Class representing a call to the python bitwise AND operator.
    I.e:
        a & b
    is equivalent to:
        PyccelBitAnd(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 10

    def __repr__(self):
        return '{} & {}'.format(self.args[0], self.args[1])
