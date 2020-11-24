"""
Module handling alll python builtin operators
These operators all have a precision as detailed here:
    https://docs.python.org/3/reference/expressions.html#operator-precedence
They also have specific rules to determine the dtype, precision, rank, shape
"""
from sympy.core.expr          import Expr

from .basic     import PyccelAstNode

from .builtins import PythonInt

from .datatypes import NativeBool, NativeInteger, NativeReal, NativeComplex, NativeString, default_precision

__all__ = (
    'PyccelOperator',
    'PyccelPow',
    'PyccelAdd',
    'PyccelMinus',
    'PyccelMul',
    'PyccelDiv',
    'PyccelMod',
    'PyccelFloorDiv',
    'PyccelEq',
    'PyccelNe',
    'PyccelLt',
    'PyccelLe',
    'PyccelGt',
    'PyccelGe',
    'PyccelAnd',
    'PyccelOr',
    'PyccelNot',
    'PyccelRShift',
    'PyccelLShift',
    'PyccelBitXor',
    'PyccelBitOr',
    'PyccelBitAnd',
    'PyccelInvert',
    'PyccelAssociativeParenthesis',
    'PyccelUnary',
    'Relational'
)

#==============================================================================
def broadcast(shape_1, shape_2):
    """ This function broadcast two shapes using numpy broadcasting rules """
    from .core      import PyccelArraySize

    a = len(shape_1)
    b = len(shape_2)
    if a>b:
        new_shape_2 = (1,)*(a-b) + tuple(shape_2)
        new_shape_1 = shape_1
    elif b>a:
        new_shape_1 = (1,)*(b-a) + tuple(shape_1)
        new_shape_2 = shape_2
    else:
        new_shape_2 = shape_2
        new_shape_1 = shape_1

    new_shape = []
    for e1,e2 in zip(new_shape_1, new_shape_2):
        if e1 == e2:
            new_shape.append(e1)
        elif e1 == 1:
            new_shape.append(e2)
        elif e2 == 1:
            new_shape.append(e1)
        elif isinstance(e1, PyccelArraySize) and isinstance(e2, PyccelArraySize):
            new_shape.append(e1)
        elif isinstance(e1, PyccelArraySize):
            new_shape.append(e2)
        elif isinstance(e2, PyccelArraySize):
            new_shape.append(e1)
        else:
            msg = 'operands could not be broadcast together with shapes {} {}'
            msg = msg.format(shape_1, shape_2)
            errors.report(msg,severity='fatal')
    return tuple(new_shape)

#==============================================================================

class PyccelOperator(Expr, PyccelAstNode):
    """
    Abstract superclass for all builtin operators.
    The __init__ function is common
    but the functions called by __init__ are specialised

    Parameters
    ----------
    args: tuple
        The arguments passed to the operator
    """

    def __init__(self, *args):
        self._args = tuple(self._handle_precedence(args))

        if self.stage == 'syntactic':
            return
        self._set_dtype()
        self._set_shape_rank()

    @property
    def precedence(self):
        """ The precedence of the operator as defined here:
            https://docs.python.org/3/reference/expressions.html#operator-precedence
        """
        return self._precedence

    def _handle_precedence(self, args):
        """
        Insert parentheses where necessary by examining the precedence of the operator
        e.g:
            PyccelMul(a,PyccelAdd(b,c))
        means:
            a*(b+c)
        so this input will give:
            PyccelMul(a, PyccelAssociativeParenthesis(PyccelAdd(b,c)))

        Parentheses are also added were they are required for clarity

        Parameters
        ----------
        args: tuple
            The arguments passed to the operator

        Results
        -------
        args: tuple
            The arguments with the parentheses inserted
        """
        precedence = [getattr(a, 'precedence', 17) for a in args]

        if min(precedence) <= self._precedence:

            new_args = []

            for i, (a,p) in enumerate(zip(args, precedence)):
                if (p < self._precedence or (p == self._precedence and i != 0)):
                    new_args.append(PyccelAssociativeParenthesis(a))
                else:
                    new_args.append(a)
            args = tuple(new_args)

        return args

#==============================================================================

class PyccelUnaryOperator(PyccelOperator):
    """ Abstract superclass representing a python
    operator with only one argument

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """

    def __init__(self, arg):
        PyccelOperator.__init__(self, arg)

    def _set_dtype(self):
        """ Sets the dtype and precision
        They are chosen to match the argument unless the class has
        a _dtype or _precision member
        """
        a = self._args[0]
        if self._dtype is None:
            self._dtype     = a.dtype
        if self._precision is None:
            self._precision = a.precision

    def _set_shape_rank(self):
        """ Sets the shape and rank
        They are chosen to match the argument unless the class has
        a _shape or _rank member
        """
        a = self._args[0]
        if self._rank is None:
            self._rank      = a.rank
        if self._shape is None:
            self._shape     = a.shape

#==============================================================================

class PyccelUnary(PyccelUnaryOperator):
    """
    Class representing a call to the python positive operator.
    I.e:
        +a
    is equivalent to:
        PyccelUnary(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    _precedence = 14
    def _handle_precedence(self, args):
        args = PyccelUnaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

#==============================================================================

class PyccelUnarySub(PyccelUnary):
    """
    Class representing a call to the python negative operator.
    I.e:
        -a
    is equivalent to:
        PyccelUnarySub(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """

#==============================================================================

class PyccelNot(PyccelUnaryOperator):
    """
    Class representing a call to the python not operator.
    I.e:
        not a
    is equivalent to:
        PyccelNot(a)

    Parameters
    ----------
    arg: PyccelAstNode
        The argument passed to the operator
    """
    _precedence = 6
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']

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
    _precedence = 14
    _dtype     = NativeInteger()

    def _set_dtype(self):
        a = self._args[0]
        if a.dtype not in (NativeInteger(), NativeBool()):
            raise TypeError('unsupported operand type(s): {}'.format(self))

        self._args      = (PythonInt(a) if a.dtype is NativeBool() else a,)

        self._precision = a.precision

#==============================================================================

class PyccelAssociativeParenthesis(PyccelUnaryOperator):
    """
    Class representing parentheses

    Parameters
    ----------
    arg: PyccelAstNode
        The argument in the PyccelAssociativeParenthesis
    """
    _precedence = 18
    def _handle_precedence(self, args):
        return args

#==============================================================================

class PyccelBinaryOperator(PyccelOperator):
    """ Abstract superclass representing a python
    operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

    def __init__(self, arg1, arg2):
        PyccelOperator.__init__(self, arg1, arg2)

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

        if strs:
            self._handle_str_type(strs)
            assert len(integers + reals + complexes) == 0
        elif complexes:
            self._handle_complex_type(complexes)
        elif reals:
            self._handle_real_type(reals)
        elif integers:
            self._handle_integer_type(integers)
        else:
            raise TypeError('cannot determine the type of {}'.format(self))


    def _handle_str_type(self, strs):
        """
        Set dtype and precision when both arguments are strings
        """
        raise TypeError("unsupported operand type(s) for /: 'str' and 'str'")

    def _handle_complex_type(self, complexes):
        """
        Set dtype and precision when the result is complex
        """
        self._dtype     = NativeComplex()
        self._precision = max(a.precision for a in complexes)

    def _handle_real_type(self, reals):
        """
        Set dtype and precision when the result is real
        """
        self._dtype     = NativeReal()
        self._precision = max(a.precision for a in reals)

    def _handle_integer_type(self, integers):
        """
        Set dtype and precision when the result is integer
        """
        self._dtype     = NativeInteger()
        self._precision = max(a.precision for a in integers)

    def _set_shape_rank(self):
        """ Sets the shape and rank

        Strings must be scalars.

        For numeric types the rank and shape is determined according
        to numpy broadcasting rules where possible
        """
        if self._dtype is NativeString():
            self._rank  = 0
            self._shape = ()
        else:
            ranks  = [a.rank for a in  self._args]
            shapes = [a.shape for a in self._args]

            if None in ranks:
                self._rank  = None
                self._shape = None

            elif all(sh is not None for tup in shapes for sh in tup):
                shape = broadcast(self._args[0].shape, self._args[1].shape)

                self._shape = shape
                self._rank  = len(shape)
            else:
                self._rank  = max(a.rank for a in self._args)
                self._shape = [None]*self._rank

    @property
    def lhs(self):
        """ First operator argument"""
        return self._args[0]

    @property
    def rhs(self):
        """ First operator argument"""
        return self._args[1]

#==============================================================================

class PyccelArithmeticOperator(PyccelBinaryOperator):
    """ Abstract superclass representing a python
    arithmetic operator

    This class is necessary to handle specific precedence
    rules for arithmetic operators
    I.e. to handle the error:
    Extension: Unary operator following arithmetic operator (use parentheses)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    def _handle_precedence(self, args):
        args = PyccelBinaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

#==============================================================================

class PyccelPow(PyccelArithmeticOperator):
    """
    Class representing a call to the python exponent operator.
    I.e:
        a ** b
    is equivalent to:
        PyccelPow(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence  = 15

#==============================================================================

class PyccelAdd(PyccelArithmeticOperator):
    """
    Class representing a call to the python addition operator.
    I.e:
        a + b
    is equivalent to:
        PyccelAdd(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 12

    def _handle_str_type(self, strs):
        self._dtype = NativeString()

#==============================================================================

class PyccelMul(PyccelArithmeticOperator):
    """
    Class representing a call to the python multiplication operator.
    I.e:
        a * b
    is equivalent to:
        PyccelMul(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 13

#==============================================================================

class PyccelMinus(PyccelAdd):
    """
    Class representing a call to the python subtraction operator.
    I.e:
        a - b
    is equivalent to:
        PyccelMinus(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

#==============================================================================

class PyccelDiv(PyccelArithmeticOperator):
    """
    Class representing a call to the python division operator.
    I.e:
        a / b
    is equivalent to:
        PyccelDiv(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 13

    def _handle_integer_type(self, integers):
        self._dtype     = NativeReal()
        self._precision = default_precision['real']

#==============================================================================

class PyccelMod(PyccelArithmeticOperator):
    """
    Class representing a call to the python modulo operator.
    I.e:
        a % b
    is equivalent to:
        PyccelMod(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 13

#==============================================================================

class PyccelFloorDiv(PyccelArithmeticOperator):
    """
    Class representing a call to the python integer division operator.
    I.e:
        a // b
    is equivalent to:
        PyccelFloorDiv(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 13

#==============================================================================

class PyccelBitOperator(PyccelBinaryOperator):
    """ Abstract superclass representing a python
    bitwise operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _rank = 0
    _shape = ()

    def _set_shape_rank(self):
        pass

    def _handle_complex_type(self, complexes):
        raise TypeError('unsupported operand type(s): {}'.format(self))

    def _handle_real_type(self, reals):
        raise TypeError('unsupported operand type(s): {}'.format(self))

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
    _precedence = 11

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
    _precedence = 11

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
    _precedence = 9

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
    _precedence = 8

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
    _precedence = 10

#==============================================================================

class PyccelComparisonOperator(PyccelBinaryOperator):
    """ Abstract superclass representing a python
    comparison operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 7
    _dtype = NativeBool()
    _precision = default_precision['bool']
    def _set_dtype(self):
        pass

#==============================================================================

class PyccelEq(PyccelComparisonOperator):
    """
    Class representing a call to the python equality operator.
    I.e:
        a == b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

class PyccelNe(PyccelComparisonOperator):
    """
    Class representing a call to the python inequality operator.
    I.e:
        a != b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

class PyccelLt(PyccelComparisonOperator):
    """
    Class representing a call to the python less than operator.
    I.e:
        a < b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

class PyccelLe(PyccelComparisonOperator):
    """
    Class representing a call to the python less or equal operator.
    I.e:
        a <= b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

class PyccelGt(PyccelComparisonOperator):
    """
    Class representing a call to the python greater than operator.
    I.e:
        a > b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

class PyccelGe(PyccelComparisonOperator):
    """
    Class representing a call to the python greater or equal operator.
    I.e:
        a >= b
    is equivalent to:
        PyccelEq(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """

#==============================================================================

class PyccelBooleanOperator(PyccelOperator):
    """ Abstract superclass representing a python
    boolean operator with two arguments

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']

    def _set_dtype(self):
        pass
    def _set_shape_rank(self):
        pass

#==============================================================================

class PyccelAnd(PyccelBooleanOperator):
    """
    Class representing a call to the python AND operator.
    I.e:
        a and b
    is equivalent to:
        PyccelAnd(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 5
    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelOr) else a for a in args)
        return args

#==============================================================================

class PyccelOr(PyccelBooleanOperator):
    """
    Class representing a call to the python OR operator.
    I.e:
        a or b
    is equivalent to:
        PyccelOr(a, b)

    Parameters
    ----------
    arg1: PyccelAstNode
        The first argument passed to the operator
    arg2: PyccelAstNode
        The second argument passed to the operator
    """
    _precedence = 4
    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelAnd) else a for a in args)
        return args

#==============================================================================

class Is(PyccelBooleanOperator):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast import Is
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> Is(x, Nil())
    Is(x, None)
    """
    _precedence = 7

#==============================================================================

class IsNot(Is):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast import IsNot
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> IsNot(x, Nil())
    IsNot(x, None)
    """

#==============================================================================

Relational = (PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe, PyccelAnd, PyccelOr,  PyccelNot, Is, IsNot)

