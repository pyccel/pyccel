#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module handling all Python builtin operators
These operators all have a precision as detailed here:
    https://docs.python.org/3/reference/expressions.html#operator-precedence
They also have specific rules to determine the dtype, rank, shape
"""
import functools

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

    def _calculate_type(self, arg):
        """
        Get the dtype.

        Arguments must be integers or booleans. Any booleans are cast
        to integers.

        Parameters
        ----------
        arg : TypedAstNode
            The argument passed to the operator.

        Returns
        -------
        DataType
            The  datatype of the result of the operation.
        """
        dtype = PythonNativeInt()
        assert isinstance(getattr(arg.dtype, 'primitive_type', None), (PyccelBooleanType, PyccelIntegerType))

        self._args      = (PythonInt(arg) if arg.dtype is PythonNativeBool() else arg,)
        return dtype

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
    __slots__ = ('_class_type',)

    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)

    def _set_order(self):
        pass

    def _calculate_type(self, *args):
        """
        Get the dtype.

        If one argument is a string then all arguments must be strings.

        If the arguments are numeric then the dtype
        matches the broadest type.
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex

        Parameters
        ----------
        *args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        DataType
            The  datatype of the result of the operation.
        """
        try:
            class_type = sum((a.class_type for a in args), start=GenericType())
        except NotImplementedError:
            raise TypeError(f'Cannot determine the type of {args}') #pylint: disable=raise-missing-from

        assert isinstance(getattr(class_type, 'primitive_type', None), (PyccelBooleanType, PyccelIntegerType))

        self._args = [PythonInt(a) if a.dtype is PythonNativeBool() else a for a in args]

        return class_type

    def _set_shape_rank(self):
        """
        Set the shape and rank of the resulting object.

        Set the shape and rank of the resulting object. For a PyccelBitOperator,
        the shape and rank are class attributes so nothing needs to be done.
        """

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
    def _calculate_type(self, *args):
        """
        Get the dtype.

        If one argument is a string then all arguments must be strings.

        If the arguments are numeric then the dtype
        matches the broadest type.
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex

        Parameters
        ----------
        *args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        DataType
            The  datatype of the result of the operation.
        """
        try:
            class_type = functools.reduce(lambda a, b: a & b, (a.class_type for a in args))
        except NotImplementedError:
            raise TypeError(f'Cannot determine the type of {args}') #pylint: disable=raise-missing-from

        primitive_type = class_type.primitive_type
        assert isinstance(primitive_type, (PyccelBooleanType, PyccelIntegerType))

        if isinstance(primitive_type, PyccelIntegerType):
            self._args = [PythonInt(a) if a.dtype is PythonNativeBool() else a for a in args]

        return class_type

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
