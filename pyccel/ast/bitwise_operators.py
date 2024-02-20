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
from .datatypes    import PyccelBooleanType, PyccelIntegerType
from .datatypes    import PythonNativeBool, PythonNativeInt, GenericType
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
        dtype = PythonNativeInt()
        assert isinstance(getattr(arg.dtype, 'primitive_type', None), (PyccelBooleanType, PyccelIntegerType))

        self._args      = (PythonInt(arg) if arg.dtype is PythonNativeBool() else arg,)
        precision = arg.precision
        return dtype, precision, dtype

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
    __slots__ = ('_dtype','_class_type')

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
        try:
            dtype = sum((a.dtype for a in args), start=GenericType())
            class_type = sum((a.class_type for a in args), start=GenericType())
        except NotImplementedError:
            raise TypeError(f'Cannot determine the type of {args}') #pylint: disable=raise-missing-from

        assert isinstance(getattr(dtype, 'primitive_type', None), (PyccelBooleanType, PyccelIntegerType))
        return dtype, class_type

    def _set_shape_rank(self):
        pass

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
        assert all(isinstance(getattr(a.dtype, 'primitive_type', None), (PyccelBooleanType, PyccelIntegerType)) for a in args)
        if all(a.dtype.primitive_type is PyccelBooleanType() for a in args):
            dtype = PythonNativeBool()
        else:
            dtype = PythonNativeInt()
        return dtype

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
