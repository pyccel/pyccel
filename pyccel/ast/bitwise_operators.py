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
from .operators    import PyccelUnaryOperator, PyccelOperator

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
    arg : TypedAstNode
        The argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 14

    def _calculate_dtype(self, arg):
        """
        Get the dtype and precision.

        Arguments must be integers or booleans. Any booleans are cast
        to integers.

        Parameters
        ----------
        arg : tuple of TypedAstNode
            The argument passed to the operator.

        Returns
        -------
        dtype : DataType
            The  datatype of the result of the operation.
        precision : integer
            The precision of the result of the operation.
        """
        dtype = NativeInteger()
        if arg.dtype not in (NativeInteger(), NativeBool()):
            raise TypeError(f'unsupported operand type(s): {arg}')

        self._args      = (PythonInt(arg) if arg.dtype is NativeBool() else arg,)
        precision = arg.precision
        return dtype, precision

    def __repr__(self):
        return f'~{repr(self.args[0])}'

#==============================================================================

class PyccelBitOperator(PyccelOperator):
    """
    Abstract superclass representing a Python bitwise operator with two arguments.

    Abstract superclass representing a Python bitwise operator with two arguments.

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    _shape = None
    _rank  = 0
    _order = None
    __slots__ = ('_dtype','_precision')

    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)

    def _set_order(self):
        pass

    def _calculate_dtype(self, *args):
        """
        Get the dtype and precision.

        If one argument is a string then all arguments must be strings.

        If the arguments are numeric then the dtype and precision
        match the broadest type and the largest precision.
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex

        Parameters
        ----------
        *args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        dtype : DataType
            The  datatype of the result of the operation.
        precision : integer
            The precision of the result of the operation.
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
        """
        Set dtype and precision when the result is an integer.

        Calculate the dtype and precision of the result from the arguments in
        the case where the result is an integer, ie. when the arguments are all
        booleans or integers.

        Parameters
        ----------
        args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        dtype : DataType
            The datatype of the result of the operator.

        precision : int
            The precision of the result of the operator.
        """
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
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
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
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
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
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    def _handle_integer_type(self, args):
        """
        Set dtype and precision when the result is an integer.

        Calculate the dtype and precision of the result from the arguments in
        the case where the result is an integer, ie. when the arguments are all
        booleans or integers.

        Parameters
        ----------
        args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        dtype : DataType
            The datatype of the result of the operator.

        precision : int
            The precision of the result of the operator.
        """
        if all(a.dtype is NativeInteger() for a in args):
            dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            dtype = NativeBool()
        else:
            dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]
        precision = max_precision(args, NativeInteger())
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
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
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
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
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
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 10

    def __repr__(self):
        return f'{self.args[0]} & {self.args[1]}'
