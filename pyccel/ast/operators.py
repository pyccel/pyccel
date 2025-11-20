#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module handling all Python builtin operators
These operators all have a precedence as detailed here:
https://docs.python.org/3/reference/expressions.html#operator-precedence
They also have specific rules to determine the dtype, rank, shape, class_type
"""

from pyccel.utilities.stage import PyccelStage

from ..errors.errors        import Errors, PyccelSemanticError

from .basic                 import TypedAstNode

from .datatypes             import PythonNativeBool, PythonNativeFloat
from .datatypes             import StringType, FixedSizeNumericType
from .datatypes             import PrimitiveBooleanType, PrimitiveIntegerType

from .literals              import Literal, LiteralInteger, LiteralFloat, LiteralComplex
from .literals              import Nil, NilArgument, LiteralTrue, LiteralFalse
from .literals              import convert_to_literal

from .numpytypes            import NumpyNDArrayType

errors = Errors()
pyccel_stage = PyccelStage()

__all__ = (
    # --- Base classes ---
    'PyccelArithmeticOperator',
    'PyccelBinaryOperator',
    'PyccelBooleanOperator',
    'PyccelComparisonOperator',
    'PyccelOperator',
    'PyccelUnaryOperator',
    # --- Operator classes ---
    'IfTernaryOperator',
    'PyccelAdd',
    'PyccelAnd',
    'PyccelAssociativeParenthesis',
    'PyccelDiv',
    'PyccelEq',
    'PyccelFloorDiv',
    'PyccelGe',
    'PyccelGt',
    'PyccelIn',
    'PyccelIs',
    'PyccelIsNot',
    'PyccelLe',
    'PyccelLt',
    'PyccelMinus',
    'PyccelMod',
    'PyccelMul',
    'PyccelNe',
    'PyccelNot',
    'PyccelOr',
    'PyccelPow',
    'PyccelUnary',
    'PyccelUnarySub',
    'Relational',
)

#==============================================================================
def broadcast(shape_1, shape_2):
    """
    Broadcast two shapes using NumPy broadcasting rules.

    Calculate the shape of the result of an operator from the shape of the arguments
    of the operator. The new shape is calculated using NumPy broadcasting rules.

    Parameters
    ----------
    shape_1 : tuple of TypedAstNode
        The shape of the first argument.
    shape_2 : tuple of TypedAstNode
        The shape of the second argument.

    Returns
    -------
    tuple of TypedAstNode
        The shape of the result of the operator.
    """

    from pyccel.ast.sympy_helper import pyccel_to_sympy #pylint:disable=cyclic-import
    if shape_1 is None and shape_2 is None:
        return None
    elif shape_1 is None:
        new_shape_1 = (LiteralInteger(1),)*len(shape_2)
        new_shape_2 = shape_2
    elif shape_2 is None:
        new_shape_1 = shape_1
        new_shape_2 = (LiteralInteger(1),)*len(shape_1)
    else:
        a = len(shape_1)
        b = len(shape_2)

        if a>b:
            new_shape_2 = (LiteralInteger(1),)*(a-b) + tuple(shape_2)
            new_shape_1 = shape_1
        elif b>a:
            new_shape_1 = (LiteralInteger(1),)*(b-a) + tuple(shape_1)
            new_shape_2 = shape_2
        else:
            new_shape_2 = shape_2
            new_shape_1 = shape_1

    new_shape = []
    for e1,e2 in zip(new_shape_1, new_shape_2):
        used_names = set()
        symbol_map = {}
        sy_e1 = pyccel_to_sympy(e1, symbol_map, used_names)
        sy_e2 = pyccel_to_sympy(e2, symbol_map, used_names)
        if sy_e1 == sy_e2:
            new_shape.append(e1)
        elif sy_e1 == 1:
            new_shape.append(e2)
        elif sy_e2 == 1:
            new_shape.append(e1)
        elif sy_e1.is_constant() and not sy_e2.is_constant():
            new_shape.append(e1)
        elif sy_e2.is_constant() and not sy_e1.is_constant():
            new_shape.append(e2)
        elif not sy_e2.is_constant() and not sy_e1.is_constant()\
                and not (sy_e1 - sy_e2).is_constant():
            new_shape.append(e1)
        else:
            shape1_code = '-'
            shape2_code = '-'
            if shape_1:
                shape1_code = ' '.join(f'{s},' for s in shape_1)
                shape1_code = f"({shape1_code})"
            if shape_2:
                shape2_code = ' '.join(f"{s}," for s in shape_2)
                shape2_code = f"({shape2_code})"
            msg = 'operands could not be broadcast together with shapes {} {}'
            msg = msg.format(shape1_code, shape2_code)
            raise PyccelSemanticError(msg)
    return tuple(new_shape)

#==============================================================================

class PyccelOperator(TypedAstNode):
    """
    Abstract superclass for all builtin operators.

    Abstract superclass for all builtin operators.
    The __init__ function is common but the functions
    called by __init__ are specialised.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the operator.
    """
    __slots__ = ('_args', )
    _attribute_nodes = ('_args',)

    @classmethod
    def make_simplified(cls, *args):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        *args : TypedAstNode
            The arguments passed to the operator.
        """
        return cls(*args)

    def __init__(self, *args):
        self._args = tuple(self._handle_precedence(args))

        if pyccel_stage == 'syntactic':
            super().__init__()
            return
        self._set_shape()
        self._set_type()
        super().__init__()

    def _set_type(self):
        """
        Set the type of the result of the operator.

        Set the class_type of the result of the operator. This function
        uses the static method `_calculate_type` to set these values. If the
        values are class parameters in a sub-class, this method must be over-ridden.
        """
        self._class_type = self._calculate_type(*self._args)  # pylint: disable=no-member

    def _set_shape(self):
        """
        Set the shape of the result of the operator.

        Set the shape of the result of the operator. This function
        uses the static method `_shape` to set these values. If the
        values are class parameters in a sub-class, this method must be overridden.
        """
        self._shape = self._calculate_shape(*self._args)  # pylint: disable=no-member

    @property
    def precedence(self):
        """
        The precedence of the operator.

        The precedence of the operator as defined here:
        https://docs.python.org/3/reference/expressions.html#operator-precedence
        The precedence shows the order in which operators should be handled.
        In this file it is represented as an integer. The higher the integer
        value of the precedence, the higher the priority of the operator.

        Returns
        -------
        int
            The precedence of the operator.
        """
        return self._precedence #pylint: disable=no-member

    def _handle_precedence(self, args):
        """
        Insert parentheses into the expression.

        Insert parentheses where necessary by examining the precedence of the operator
        e.g:
            PyccelMul(a,PyccelAdd(b,c))
        means:
            a*(b+c)
        so this input will give:
            PyccelMul(a, PyccelAssociativeParenthesis(PyccelAdd(b,c)))

        Parentheses are also added were they are required for clarity.

        Parameters
        ----------
        args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        tuple
            The arguments with the parentheses inserted.
        """
        precedence = [getattr(a, 'precedence', 17) for a in args]

        if min(precedence) <= self.precedence:

            new_args = []

            for i, (a,p) in enumerate(zip(args, precedence)):
                if (p < self.precedence or (p == self.precedence and i != 0)):
                    new_args.append(PyccelAssociativeParenthesis(a))
                else:
                    new_args.append(a)
            args = tuple(new_args)

        return args

    def __str__(self):
        return repr(self)

    @property
    def args(self):
        """ Arguments of the operator
        """
        return self._args

#==============================================================================

class PyccelUnaryOperator(PyccelOperator):
    """
    Superclass representing an operator with only one argument.

    Abstract superclass representing a Python operator with only
    one argument.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the operator.
    """
    __slots__ = ('_shape','_class_type')

    def __init__(self, arg):
        super().__init__(arg)

    @staticmethod
    def _calculate_type(arg):
        """
        Calculate the dtype and class type of the result.

        Calculate the dtype and class type of the result.
        These are equivalent to the dtype and class type
        of the only argument.

        Parameters
        ----------
        arg : TypedAstNode
            The argument passed to the operator.

        Returns
        -------
        DataType
            The Python type of the object.
        """
        return arg.class_type

    @staticmethod
    def _calculate_shape(arg):
        """
        Calculate the shape.

        Calculate the shape. It is chosen to match the argument.

        Parameters
        ----------
        arg : TypedAstNode
            The argument passed to the operator.

        Returns
        -------
        tuple[TypedAstNode]
            The shape of the resulting object.
        """
        return arg.shape

#==============================================================================

class PyccelUnary(PyccelUnaryOperator):
    """
    Class representing a call to the Python positive operator.

    Class representing a call to the Python positive operator.
    I.e:

        +a

    is equivalent to:
    >>> PyccelUnary(a)

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 14
    def _handle_precedence(self, args):
        args = PyccelUnaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

    def __repr__(self):
        return f'+{repr(self.args[0])}'

#==============================================================================

class PyccelUnarySub(PyccelUnary):
    """
    Class representing a call to the Python negative operator.

    Class representing a call to the Python negative operator.
    I.e:

        -a

    is equivalent to:
    >>> PyccelUnarySub(a)

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the operator.
    """
    __slots__ = ()

    def __repr__(self):
        return f'-{repr(self.args[0])}'

    def __index__(self):
        return -int(self.args[0])

#==============================================================================

class PyccelNot(PyccelUnaryOperator):
    """
    Class representing a call to the Python not operator.

    Class representing a call to the Python not operator.
    I.e:

        not a

    is equivalent to:
    >>> PyccelNot(a).

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 6
    _class_type = PythonNativeBool()


    @classmethod
    def make_simplified(cls, arg):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        arg : TypedAstNode
            The argument passed to the operator.
        """
        if isinstance(arg, PyccelEq):
            arg1, arg2 = arg.args
            return PyccelNe(arg1, arg2)
        elif isinstance(arg, PyccelNe):
            arg1, arg2 = arg.args
            return PyccelEq(arg1, arg2)
        else:
            return cls(arg)

    def _set_type(self):
        """
        Set the type of the result of the operator.

        Set the class_type of the result of the operator. Nothing needs
        to be done here as the type is a class variable.
        """

    @staticmethod
    def _calculate_shape(arg):
        """
        Calculate the shape.

        Calculate the shape. It is chosen to match the argument.

        Parameters
        ----------
        arg : TypedAstNode
            The argument passed to the operator.

        Returns
        -------
        tuple[TypedAstNode]
            The shape of the resulting object.
        """
        return None

    def __repr__(self):
        return f'not {repr(self.args[0])}'

#==============================================================================

class PyccelAssociativeParenthesis(PyccelUnaryOperator):
    """
    Class representing parentheses.

    Class representing parentheses around an expression to group
    ideas or to ensure the execution order of the code.

    Parameters
    ----------
    arg : TypedAstNode
        The argument in the PyccelAssociativeParenthesis.
    """
    __slots__ = () # ok
    _precedence = 18
    def _handle_precedence(self, args):
        return args

    def __repr__(self):
        return f'({repr(self.args[0])})'

#==============================================================================

class PyccelBinaryOperator(PyccelOperator):
    """
    Superclass representing a Python operator with two arguments.

    Abstract superclass representing a Python operator with two
    arguments.

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ('_shape','_class_type')

    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)

    @classmethod
    def _calculate_type(cls, arg1, arg2):
        """
        Sets the dtype and class type.

        If one argument is a string then all arguments must be strings

        If the arguments are numeric then the dtype and class type
        match the broadest type
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.

        Returns
        -------
        DataType
            The Python type of the object.

        Raises
        ------
        TypeError
            Raised if the new type cannot be deduced by checking the __add__ operator
            of the class types.
        """
        try:
            return arg1.class_type + arg2.class_type
        except NotImplementedError:
            raise TypeError(f'Cannot determine the type of ({arg1}, {arg2})') #pylint: disable=raise-missing-from

    @staticmethod
    def _calculate_shape(arg1, arg2):
        """
        Calculate the shape.

        Strings must be scalars.

        For numeric types the shape is determined according
        to NumPy broadcasting rules where possible.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.

        Returns
        -------
        tuple[TypedAstNode]
            The shape of the resulting object.
        """
        args = (arg1, arg2)
        strs = [a for a in args if isinstance(a.dtype, StringType)]
        if strs:
            other = [a for a in args if isinstance(a.dtype, FixedSizeNumericType)]
            assert len(other) == 0
            shape = None
        elif any(isinstance(a.class_type, NumpyNDArrayType) for a in (arg1, arg2)):
            shape = broadcast(arg1.shape, arg2.shape)
        else:
            shape = None
        return shape

#==============================================================================

class PyccelArithmeticOperator(PyccelBinaryOperator):
    """
    Abstract superclass representing a Python arithmetic operator.

    This class is necessary to handle specific precedence
    rules for arithmetic operators.
    I.e. to handle the error:
    Extension: Unary operator following arithmetic operator (use parentheses).

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    def _handle_precedence(self, args):
        args = PyccelBinaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

#==============================================================================

class PyccelPow(PyccelArithmeticOperator):
    """
    Class representing a call to the Python exponent operator.

    Class representing a call to the Python exponent operator.
    I.e:

        a ** b

    is equivalent to:
    >>>  PyccelPow(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence  = 15

    def __repr__(self):
        return f'{self.args[0]} ** {self.args[1]}'

    def _handle_precedence(self, args):
        precedence = [getattr(a, 'precedence', 17) for a in args]

        if min(precedence) <= self._precedence:

            new_args = []

            for i, (a,p) in enumerate(zip(args, precedence)):
                if (p < self._precedence or (p == self._precedence and i != 1)):
                    new_args.append(PyccelAssociativeParenthesis(a))
                else:
                    new_args.append(a)
            args = tuple(new_args)

        return args

#==============================================================================

class PyccelAdd(PyccelArithmeticOperator):
    """
    Class representing a call to the Python addition operator.

    Class representing a call to the Python addition operator.
    I.e:

        a + b

    is equivalent to:
    >>> PyccelAdd(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 12

    @classmethod
    def make_simplified(cls, arg1, arg2):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.
        """
        if isinstance(arg2, PyccelUnarySub):
            return PyccelMinus.make_simplified(arg1, arg2.args[0])

        class_type = cls._calculate_type(arg1, arg2)

        if isinstance(arg1, Literal) and isinstance(arg2, Literal):
            return convert_to_literal(arg1.python_value + arg2.python_value,
                                      class_type)
        if class_type == arg2.class_type and arg1 == 0:
            return arg2
        if class_type == arg1.class_type and arg2 == 0:
            return arg1

        if isinstance(arg1, PyccelMinus) and arg1.args[1] == arg2 \
                and arg1.args[0].class_type == class_type:
            return arg1.args[0]
        if isinstance(arg1, PyccelAdd) and isinstance(arg1.args[1], Literal) and isinstance(arg2, Literal):
            return PyccelAdd(arg1.args[0], PyccelAdd.make_simplified(arg1.args[1], arg2))

        return cls(arg1, arg2)

    def __new__(cls, arg1, arg2):
        if isinstance(arg1, (LiteralInteger, LiteralFloat)) and \
            isinstance(arg2, LiteralComplex) and \
            arg2.real == LiteralFloat(0):
            return LiteralComplex(arg1, arg2.imag)
        elif isinstance(arg2, (LiteralInteger, LiteralFloat)) and \
            isinstance(arg1, LiteralComplex) and \
            arg1.real == LiteralFloat(0):
            return LiteralComplex(arg2, arg1.imag)
        else:
            return super().__new__(cls)

    @classmethod
    def _calculate_type(cls, arg1, arg2):
        """
        Sets the dtype and class type.

        If one argument is a string then all arguments must be strings

        If the arguments are numeric then the dtype and class type
        match the broadest type
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.

        Returns
        -------
        DataType
            The Python type of the object.
        """
        if arg1.dtype == arg2.dtype == StringType():
            return arg1.dtype
        else:
            return super()._calculate_type(arg1, arg2)

    def __repr__(self):
        return f'{self.args[0]} + {self.args[1]}'

#==============================================================================

class PyccelMul(PyccelArithmeticOperator):
    """
    Class representing a call to the Python multiplication operator.

    Class representing a call to the Python multiplication operator.
    I.e:

        a * b

    is equivalent to:
    >>> PyccelMul(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 13

    @classmethod
    def make_simplified(cls, arg1, arg2):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.
        """
        class_type = cls._calculate_type(arg1, arg2)

        if arg1 == 1 and arg2.class_type == class_type:
            return arg2
        if arg2 == 1 and arg2.class_type == class_type:
            return arg1
        if (arg1 == 0 or arg2 == 0):
            return convert_to_literal(0, class_type)
        if (isinstance(arg1, PyccelUnarySub) and arg1.args[0] == 1) \
                and arg2.class_type == class_type:
            return PyccelUnarySub(arg2)
        if (isinstance(arg2, PyccelUnarySub) and arg2.args[0] == 1) \
                and arg1.class_type == class_type:
            return PyccelUnarySub(arg1)
        if isinstance(arg1, Literal) and isinstance(arg2, Literal):
            return convert_to_literal(arg1.python_value * arg2.python_value,
                                      class_type)
        return cls(arg1, arg2)

    def __repr__(self):
        return f'{repr(self.args[0])} * {repr(self.args[1])}'

#==============================================================================

class PyccelMinus(PyccelArithmeticOperator):
    """
    Class representing a call to the Python subtraction operator.

    Class representing a call to the Python subtraction operator.
    I.e:

        a - b

    is equivalent to:
    >>> PyccelMinus(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 12

    @classmethod
    def make_simplified(cls, arg1, arg2):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.
        """
        if isinstance(arg2, PyccelUnarySub):
            return PyccelAdd.make_simplified(arg1, arg2.args[0])
        elif isinstance(arg1, Literal) and isinstance(arg2, Literal):
            dtype = cls._calculate_type(arg1, arg2)
            return convert_to_literal(arg1.python_value - arg2.python_value,
                                      dtype)

        class_type = cls._calculate_type(arg1, arg2)

        if class_type == arg2.class_type and arg1 == 0:
            return PyccelUnarySub(arg2)
        if class_type == arg1.class_type and arg2 == 0:
            return arg1
        if isinstance(arg1, PyccelAdd) and arg1.args[1] == arg2 \
                and arg1.args[0].class_type == class_type:
            return arg1.args[0]
        if isinstance(arg1, PyccelAdd) and isinstance(arg1.args[1], Literal) and isinstance(arg2, Literal):
            return PyccelAdd(arg1.args[0], PyccelMinus.make_simplified(arg1.args[1], arg2))
        if isinstance(arg1, PyccelMinus) and isinstance(arg1.args[1], Literal) and isinstance(arg2, Literal):
            return PyccelMinus(arg1.args[0], PyccelAdd.make_simplified(arg1.args[1], arg2))

        return cls(arg1, arg2)

    def __new__(cls, arg1 = None, arg2 = None):
        if isinstance(arg1, LiteralFloat) and \
            isinstance(arg2, LiteralComplex) and \
            arg2.real == LiteralFloat(0):
            return LiteralComplex(arg1, -arg2.imag.python_value)
        elif isinstance(arg2, LiteralFloat) and \
            isinstance(arg1, LiteralComplex) and \
            arg1.real == LiteralFloat(0):
            return LiteralComplex(-arg2.python_value, arg1.imag)
        else:
            return super().__new__(cls)

    def __repr__(self):
        return f'{repr(self.args[0])} - {repr(self.args[1])}'

#==============================================================================

class PyccelDiv(PyccelArithmeticOperator):
    """
    Class representing a call to the Python division operator.

    Class representing a call to the Python division operator.
    I.e:

        a / b

    is equivalent to:
    >>> PyccelDiv(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 13

    @classmethod
    def make_simplified(cls, arg1, arg2):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.
        """
        class_type = cls._calculate_type(arg1, arg2)
        if arg2 == 1 and arg1.class_type == class_type:
            return arg1
        if isinstance(arg1, Literal) and isinstance(arg2, Literal):
            return convert_to_literal(arg1.python_value / arg2.python_value,
                                      class_type)

        return cls(arg1, arg2)

    @classmethod
    def _calculate_type(cls, arg1, arg2):
        """
        Sets the dtype and class type.

        If one argument is a string then all arguments must be strings

        If the arguments are numeric then the dtype and class type
        match the broadest type
        e.g.
            1 + 2j -> PyccelAdd(LiteralInteger, LiteralComplex) -> complex

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.

        Returns
        -------
        DataType
            The Python type of the object.
        """
        class_type = super()._calculate_type(arg1, arg2)

        if class_type.primitive_type in (PrimitiveIntegerType(), PrimitiveBooleanType()):
            class_type = class_type.switch_basic_type(PythonNativeFloat())

        return class_type

    def __repr__(self):
        return f'{repr(self.args[0])} / {repr(self.args[1])}'

#==============================================================================

class PyccelMod(PyccelArithmeticOperator):
    """
    Class representing a call to the Python modulo operator.

    Class representing a call to the Python modulo operator.
    I.e:

        a % b

    is equivalent to:
    >>> PyccelMod(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 13

    def __repr__(self):
        return f'{repr(self.args[0])} % {repr(self.args[1])}'

#==============================================================================

class PyccelFloorDiv(PyccelArithmeticOperator):
    """
    Class representing a call to the Python integer division operator.

    Class representing a call to the Python integer division operator.
    I.e:

        a // b

    is equivalent to:
    >>> PyccelFloorDiv(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 13

    def __repr__(self):
        return f'{repr(self.args[0])} // {repr(self.args[1])}'

#==============================================================================

class PyccelComparisonOperator(PyccelBinaryOperator):
    """
    Superclass representing a Python comparison operator.

    Abstract superclass representing a Python comparison
    operator with two arguments.

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 7

    @classmethod
    def _calculate_type(cls, arg1, arg2):
        """
        Calculate the dtype and class type of the result.

        Calculate the dtype and class type of the result.
        These are the dtype and class type which represent a
        boolean.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.

        Returns
        -------
        dtype : DataType
            The underlying datatype of the object.
        class_type : DataType
            The Python type of the object.
        """
        dtype = PythonNativeBool()
        possible_class_types = set(a.class_type for a in (arg1, arg2) \
                        if isinstance(a.class_type, NumpyNDArrayType))
        if len(possible_class_types) == 0:
            class_type = dtype
        elif len(possible_class_types) == 1:
            class_type = possible_class_types.pop().switch_basic_type(dtype)
        else:
            description = f"({arg1!r} {cls.op} {arg2!r})" # pylint: disable=no-member
            raise NotImplementedError("Can't deduce type for comparison operator"
                                      f" with multiple containers {description}")
        return class_type

    def __repr__(self):
        return f'{repr(self.args[0])} {self.op} {repr(self.args[1])}' # pylint: disable=no-member

#==============================================================================

class PyccelEq(PyccelComparisonOperator):
    """
    Class representing a call to the Python equality operator.

    Class representing a call to the Python equality operator.
    I.e:

        a == b

    is equivalent to:
    >>> PyccelEq(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    op = "=="

    def __new__(cls, arg1 = None, arg2 = None):
        if isinstance(arg1, Nil) or isinstance(arg2, Nil):
            return PyccelIs(arg1, arg2)
        else:
            return super().__new__(cls)

class PyccelNe(PyccelComparisonOperator):
    """
    Class representing a call to the Python inequality operator.

    Class representing a call to the Python inequality operator.
    I.e:

        a != b

    is equivalent to:
    >>> PyccelNe(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    op = "!="

    def __new__(cls, arg1 = None, arg2 = None):
        if isinstance(arg1, Nil) or isinstance(arg2, Nil):
            return PyccelIsNot(arg1, arg2)
        else:
            return super().__new__(cls)

class PyccelLt(PyccelComparisonOperator):
    """
    Class representing a call to the Python less than operator.

    Class representing a call to the Python less than operator.
    I.e:

        a < b

    is equivalent to:
    >>> PyccelLt(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    op = "<"

class PyccelLe(PyccelComparisonOperator):
    """
    Class representing a call to the Python less or equal operator.

    Class representing a call to the Python less or equal operator.
    I.e:

        a <= b

    is equivalent to:
    >>> PyccelLe(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    op = "<="

class PyccelGt(PyccelComparisonOperator):
    """
    Class representing a call to the Python greater than operator.

    Class representing a call to the Python greater than operator.
    I.e:

        a > b

    is equivalent to:
    >>> PyccelGt(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    op = ">"

    @classmethod
    def make_simplified(cls, arg1, arg2):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        arg1 : TypedAstNode
            The first argument passed to the operator.
        arg2 : TypedAstNode
            The second argument passed to the operator.
        """

        if all(isinstance(a, Literal) or isinstance(a, PyccelUnarySub) and isinstance(a.args[0], Literal)
                for a in (arg1, arg2)):
            arg1_val = arg1.python_value if isinstance(arg1, Literal) else -arg1.args[0].python_value
            arg2_val = arg2.python_value if isinstance(arg2, Literal) else -arg2.args[0].python_value
            return convert_to_literal(arg1_val > arg2_val)
        else:
            return cls(arg1, arg2)

class PyccelGe(PyccelComparisonOperator):
    """
    Class representing a call to the Python greater or equal operator.

    Class representing a call to the Python greater or equal operator.
    I.e:

        a >= b

    is equivalent to:
    >>> PyccelGe(a, b)

    Parameters
    ----------
    arg1 : TypedAstNode
        The first argument passed to the operator.
    arg2 : TypedAstNode
        The second argument passed to the operator.
    """
    __slots__ = ()
    op = ">="

#==============================================================================

class PyccelBooleanOperator(PyccelOperator):
    """
    Superclass representing a boolean operator with two arguments.

    Abstract superclass representing a Python
    boolean operator with two arguments.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the operator.
    """
    _shape = None
    _class_type = PythonNativeBool()

    __slots__ = ()

    def _set_type(self):
        """
        Set the type of the result of the operator.

        Set the class_type of the result of the operator. Nothing needs
        to be done here as the type is a class variable.
        """

    def _set_shape(self):
        """
        Set the shape of the result of the operator.

        Set the shape of the result of the operator. Nothing needs
        to be done here as the shape is a class variable.
        """

#==============================================================================

class PyccelAnd(PyccelBooleanOperator):
    """
    Class representing a call to the Python AND operator.

    Class representing a call to the Python AND operator.
    I.e:

        a and b

    is equivalent to:
    >>> PyccelAnd(a, b)

    Parameters
    ----------
    *args : TypedAstNode
        The arguments passed to the operator.
    """
    __slots__ = ()
    _precedence = 5

    @classmethod
    def make_simplified(cls, *args):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        *args : TypedAstNode
            The arguments passed to the operator.
        """
        if any(isinstance(a, LiteralFalse) for a in args):
            return LiteralFalse()
        if all(isinstance(a, LiteralTrue) for a in args):
            return LiteralTrue()
        args = tuple(a for a in args if not isinstance(a, LiteralTrue))
        return cls(*args)

    def __init__(self, *args):
        args = tuple(ai for a in args for ai in (a.args if isinstance(a, PyccelAnd) else [a]))
        super().__init__(*args)


    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelOr) else a for a in args)
        return args

    def __repr__(self):
        return ' and '.join(repr(a) for a in self.args)

#==============================================================================

class PyccelOr(PyccelBooleanOperator):
    """
    Class representing a call to the Python OR operator.

    Class representing a call to the Python OR operator.
    I.e:

        a or b

    is equivalent to:
    >>> PyccelOr(a, b)

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the operator.
    """
    __slots__ = ()
    _precedence = 4

    @classmethod
    def make_simplified(cls, *args):
        """
        Call the class constructor after making any simplifications to the expression.

        Call the class constructor after making any simplifications to the expression.
        This method should be overridden by sub-classes.

        Parameters
        ----------
        *args : TypedAstNode
            The arguments passed to the operator.
        """
        if any(isinstance(a, LiteralTrue) for a in args):
            return LiteralTrue()
        elif all(isinstance(a, LiteralFalse) for a in args):
            return LiteralFalse()
        args = tuple(a for a in args if not isinstance(a, LiteralFalse))
        return cls(*args)

    def __init__(self, *args):
        args = tuple(ai for a in args for ai in (a.args if isinstance(a, PyccelOr) else [a]))
        super().__init__(*args)

    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelAnd) else a for a in args)
        return args

    def __repr__(self):
        return ' or '.join(repr(a) for a in self.args)

#==============================================================================

class PyccelIs(PyccelBooleanOperator):
    """
    Represents an `is` expression in the code.

    Represents an `is` expression in the code.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the operator.

    Examples
    --------
    >>> from pyccel.ast.operators import PyccelIs
    >>> from pyccel.ast.literals  import Nil
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> x = PyccelSymbol('x')
    >>> PyccelIs(x, Nil())
    PyccelIs(x, None)
    """
    __slots__ = ()
    _precedence = 7

    @property
    def lhs(self):
        """ First operator argument"""
        return self._args[0]

    @property
    def rhs(self):
        """ First operator argument"""
        return self._args[1]

    def __repr__(self):
        return f'{repr(self.args[0])} is {repr(self.args[1])}'

    def eval(self):
        """ Determines the value of the expression `x is None` when `x` is known.

        If a boolean value cannot be computed, return the string "unknown".
        """
        # evaluate `x is None` when x = None
        if self.rhs is Nil() and isinstance(self.lhs, NilArgument):
            return True
        # evaluate `x is not None` when x is known and different to None
        elif self.rhs is Nil() and not getattr(self.lhs, 'self.lhs.is_optional', False):
            return False
        # The result of the expression is unknown if the rhs is not None
        # or the lhs is an  optional variable
        else:
            return "unknown"

#==============================================================================

class PyccelIsNot(PyccelIs):
    """
    Represents a `is not` expression in the code.

    Represents a `is not` expression in the code.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the operator.

    Examples
    --------
    >>> from pyccel.ast.operators import PyccelIsNot
    >>> from pyccel.ast.literals  import Nil
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> x = PyccelSymbol('x')
    >>> PyccelIsNot(x, Nil())
    PyccelIsNot(x, None)
    """
    __slots__ = ()

    def __repr__(self):
        return f'{repr(self.args[0])} is not {repr(self.args[1])}'

    def eval(self):
        """ Determines the value of the expression `x is not None` when `x` is known.

        If a boolean value cannot be computed, return the string "unknown".
        """
        # evaluate `x is not None` when x = None
        if self.rhs is Nil() and isinstance(self.lhs, NilArgument):
            return False
        # evaluate `x is not None` when x is known and different to None
        elif self.rhs is Nil() and not getattr(self.lhs, 'self.lhs.is_optional', False):
            return True
        # The result of the expression is unknown if the rhs is not None
        # or the lhs is an  optional variable
        else:
            return "unknown"

#==============================================================================

class PyccelIn(PyccelBooleanOperator):
    """
    Represents an `in` expression in the code.

    Represents an `in` expression in the code.

    Parameters
    ----------
    element : TypedAstNode
        The first argument passed to the operator.

    container : TypedAstNode
        The first argument passed to the operator.
    """
    __slots__ = ()
    _precedence = 7

    def __init__(self, element, container):
        super().__init__(element, container)

    @property
    def element(self):
        """
        First operator argument.

        First operator argument.
        """
        return self._args[0]

    @property
    def container(self):
        """
        Second operator argument.

        Second operator argument.
        """
        return self._args[1]

#==============================================================================

class IfTernaryOperator(PyccelOperator):
    """
    Represent a ternary conditional operator in the code.

    Represent a ternary conditional operator in the code,
    of the form (a if cond else b).

    Parameters
    ----------
    cond : TypedAstNode
        The condition which determines which result is returned.
    value_true : TypedAstNode
        The value returned if the condition is true.
    value_false : TypedAstNode
        The value returned if the condition is false.

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.operators import IfTernaryOperator
    >>> n = PyccelSymbol('n')
    >>> x = 5 if n > 1 else 2
    >>> IfTernaryOperator(PyccelGt(n > 1),  5,  2)
    IfTernaryOperator(PyccelGt(n > 1),  5,  2)
    """
    __slots__ = ('_shape','_class_type')
    _precedence = 3

    def __init__(self, cond, value_true, value_false):
        super().__init__(cond, value_true, value_false)

        if pyccel_stage == 'syntactic':
            return
        if isinstance(value_true , Nil) or isinstance(value_false, Nil):
            errors.report('None is not implemented for Ternary Operator', severity='fatal')
        if value_true.dtype != value_false.dtype:
            if not (isinstance(value_true.dtype, FixedSizeNumericType) and isinstance(value_false.dtype, FixedSizeNumericType)):
                errors.report('The types are incompatible in IfTernaryOperator', severity='fatal')
        if value_false.rank != value_true.rank :
            errors.report('Ternary Operator results should have the same rank', severity='fatal')
        if value_false.shape != value_true.shape :
            errors.report('Ternary Operator results should have the same shape', severity='fatal')

    @staticmethod
    def _calculate_type(cond, value_true, value_false):
        """
        Calculate the dtype and class type of the result.

        Calculate the dtype and class type of the result. The dtype,
        and class type are calculated from the types of the values if
        true or false.

        Parameters
        ----------
        cond : TypedAstNode
            The first argument passed to the operator representing the condition.
        value_true : TypedAstNode
            The second argument passed to the operator representing the result if the
            condition is true.
        value_false : TypedAstNode
            The third argument passed to the operator representing the result if the
            condition is false.

        Returns
        -------
        DataType
            The Python type of the object.
        """
        if value_true.class_type is value_false.class_type:
            return value_true.class_type

        try:
            class_type = value_true.class_type + value_false.class_type
        except NotImplementedError:
            raise TypeError(f'Cannot determine the type of ({value_true}, {value_false})') #pylint: disable=raise-missing-from

        if value_false.class_type.order != value_true.class_type.order :
            errors.report('Ternary Operator results should have the same order', severity='fatal')

        return class_type

    @staticmethod
    def _calculate_shape(cond, value_true, value_false):
        """
        Calculate the shape of the result.

        Calculate the shape of the result of the IfTernaryOperator.
        The shape is equal to the shape of one of the outputs.

        Parameters
        ----------
        cond : TypedAstNode
            The first argument passed to the operator representing the condition.
        value_true : TypedAstNode
            The second argument passed to the operator representing the result if the
            condition is true.
        value_false : TypedAstNode
            The third argument passed to the operator representing the result if the
            condition is false.

        Returns
        -------
        tuple[TypedAstNode]
            The shape of the resulting object.
        """
        return value_true.shape

    @property
    def cond(self):
        """
        The condition property for IfTernaryOperator class
        """
        return self._args[0]

    @property
    def value_true(self):
        """
        The value_if_cond_true property for IfTernaryOperator class
        """
        return self._args[1]

    @property
    def value_false(self):
        """
        The value_if_cond_false property for IfTernaryOperator class
        """
        return self._args[2]



#==============================================================================
Relational = (PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe, PyccelAnd, PyccelOr,  PyccelNot, PyccelIs, PyccelIsNot)
