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
from .builtins     import PythonInt
from .datatypes    import (NativeBool, NativeInteger, NativeFloat,
                           NativeComplex, NativeString, NativeGeneric)
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
    Class representing a call to the python bitwise not operator.
    I.e:
        ~a
    is equivalent to:
        PyccelInvert(a)

    Parameters
    ----------
    arg: TypedAstNode
        The argument passed to the operator
    """
    __slots__ = ()
    _precedence = 14

    def _calculate_dtype(self, *args):
        dtype = NativeInteger()
        a = args[0]
        if a.dtype not in (NativeInteger(), NativeBool()):
            raise TypeError('unsupported operand type(s): {}'.format(args))

        self._args      = (PythonInt(a) if a.dtype is NativeBool() else a,)
        precision = a.precision
        return dtype, precision, dtype

    def __repr__(self):
        return '~{}'.format(repr(self.args[0]))

#==============================================================================

class PyccelBitOperator(PyccelOperator):
    """ Abstract superclass representing a python
    bitwise operator with two arguments

    Parameters
    ----------
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
        The second argument passed to the operator
    """
    _shape = None
    _rank  = 0
    _order = None
    __slots__ = ('_dtype','_precision','_class_type')

    def _set_order(self):
        pass

    def _calculate_dtype(self, *args):
        """ Sets the dtype and precision

        If one argument is a string then all arguments must be strings

        If the arguments are numeric then the dtype and precision
        match the broadest type and the largest precision
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex
        """
        try:
            dtype = sum((a.dtype for a in args), start=NativeGeneric())
            class_type = sum((a.class_type for a in args), start=NativeGeneric())
        except NotImplementedError:
            raise TypeError(f'Cannot determine the type of {args}') #pylint: disable=raise-missing-from

        if dtype in (NativeString(), NativeComplex(), NativeFloat()):
            raise TypeError('unsupported operand type(s): {}'.format(args))
        elif (dtype in (NativeInteger(), NativeBool())):
            if class_type is NativeBool():
                class_type = NativeInteger()
            return *self._handle_integer_type(args), class_type
        else:
            raise TypeError(f'Cannot determine the type of {args}')

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
    Class representing a call to the python right shift operator.
    I.e:
        a >> b
    is equivalent to:
        PyccelRShift(a, b)

    Parameters
    ----------
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
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
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
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
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
        The second argument passed to the operator
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
    Class representing a call to the python bitwise XOR operator.
    I.e:
        a ^ b
    is equivalent to:
        PyccelBitXor(a, b)

    Parameters
    ----------
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
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
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
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
    arg1: TypedAstNode
        The first argument passed to the operator
    arg2: TypedAstNode
        The second argument passed to the operator
    """
    __slots__ = ()
    _precedence = 10

    def __repr__(self):
        return '{} & {}'.format(self.args[0], self.args[1])
