#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module handling all Python builtin operators
These operators all have a precision as detailed here:
    https://docs.python.org/3/reference/expressions.html#operator-precedence
They also have specific rules to determine the dtype, precision, rank, shape
"""
from .builtins     import PythonInt
from .datatypes    import (NativeBool, NativeInteger, NativeFloat,
                           NativeComplex, NativeString)
from .internals    import max_precision
from .operators    import PyccelUnaryOperator, PyccelOperator, PyccelBinaryOperator

__all__ = (
    'PyccelBitComparisonOperator',
    'PyccelBitOperator',
    'PyccelBitAnd',
    'PyccelBitOr',
    'PyccelBitXor',
    'PyccelInvert',
    'PyccelLShift',
    'PyccelRShift',
)

#==============================================================================

class PyccelInvert(PyccelUnaryOperator):
    """
    Class representing a call to the Python bitwise not operator.

    Class representing a call to the Python bitwise not operator.
    I.e:
        ~a
    is equivalent to:
        PyccelInvert(a)

    Parameters
    ----------
    arg : PyccelAstNode
        The argument passed to the operator
    """
    __slots__ = ()
    _precedence = 14

    def _calculate_dtype(self, *args):
        dtype = NativeInteger()
        a = args[0]
        if a.dtype not in (NativeInteger(), NativeBool()):
            raise TypeError(f'unsupported operand type(s): {args}')

        self._args      = (PythonInt(a) if a.dtype is NativeBool() else a,)
        precision = a.precision
        return dtype, precision

    def __repr__(self):
        return f'~{repr(self.args[0])}'

#==============================================================================

class PyccelBitOperator(PyccelBinaryOperator):
    """
    Abstract superclass representing a Python bitwise operator with two arguments.

    Abstract superclass representing a Python bitwise operator with two arguments.

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    _shape = None
    _rank  = 0
    _order = None
    __slots__ = ('_dtype','_precision')

    def _set_order(self):
        pass

    def _calculate_dtype(self, *args):
        """
        Sets the dtype and precision.

        If one argument is a string then all arguments must be strings.

        If the arguments are numeric then the dtype and precision
        match the broadest type and the largest precision.
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex
        """
        integers  = [a for a in args if a.dtype in (NativeInteger(),NativeBool())]
        floats    = [a for a in args if a.dtype is NativeFloat()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        strs      = [a for a in args if a.dtype is NativeString()]

        if strs or complexes or floats:
            raise TypeError(f'unsupported operand type(s): {args}')
        elif integers:
            return self._handle_integer_type(integers)
        else:
            raise TypeError(f'cannot determine the type of {args}')

    def _set_shape_rank(self):
        pass

    def _handle_integer_type(self, args):
        dtype    = NativeInteger()
        integers = [a for a in args if a.dtype is NativeInteger()]

        if not integers:
            precision = -1
        else:
            precision = max_precision(integers)

        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]
        return dtype, precision

#==============================================================================

class PyccelRShift(PyccelBitOperator):
    """
    Class representing a call to the Python right shift operator.

    Class representing a call to the Python right shift operator.
    I.e:
        a >> b
    is equivalent to:
        PyccelRShift(a, b)

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 11

    def __repr__(self):
        return f'{self.args[0]} >> {self.args[1]}'

#==============================================================================

class PyccelLShift(PyccelBitOperator):
    """
    Class representing a call to the Python right shift operator.

    Class representing a call to the Python right shift operator.
    I.e:
        a << b
    is equivalent to:
        PyccelRShift(a, b)

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 11

    def __repr__(self):
        return f'{self.args[0]} << {self.args[1]}'

#==============================================================================

class PyccelBitComparisonOperator(PyccelBitOperator):
    """
    Abstract superclass representing a bitwise comparison operator.

    Abstract superclass representing a Python bitwise comparison
    operator with two arguments

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    def _handle_integer_type(self, integers):
        if all(a.dtype is NativeInteger() for a in integers):
            dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in integers):
            dtype = NativeBool()
        else:
            dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in integers]
        precision = max_precision(integers, NativeInteger())
        return dtype, precision

#==============================================================================

class PyccelBitXor(PyccelBitComparisonOperator):
    """
    Class representing a call to the Python bitwise XOR operator.

    Class representing a call to the Python bitwise XOR operator.
    I.e:
        a ^ b
    is equivalent to:
        PyccelBitXor(a, b)

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 9

    def __repr__(self):
        return f'{self.args[0]} ^ {self.args[1]}'

#==============================================================================

class PyccelBitOr(PyccelBitComparisonOperator):
    """
    Class representing a call to the Python bitwise OR operator.

    Class representing a call to the Python bitwise OR operator.
    I.e:
        a | b
    is equivalent to:
        PyccelBitOr(a, b)

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 8

    def __repr__(self):
        return f'{self.args[0]} | {self.args[1]}'

#==============================================================================

class PyccelBitAnd(PyccelBitComparisonOperator):
    """
    Class representing a call to the Python bitwise AND operator.

    Class representing a call to the Python bitwise AND operator.
    I.e:
        a & b
    is equivalent to:
        PyccelBitAnd(a, b)

    Parameters
    ----------
    arg1 : PyccelAstNode
        The first argument passed to the operator.
    arg2 : PyccelAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 10

    def __repr__(self):
        return f'{self.args[0]} & {self.args[1]}'
