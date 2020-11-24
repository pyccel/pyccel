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


class PyccelOperator(Expr, PyccelAstNode):

    def __init__(self, *args):
        self._args = tuple(self._handle_precedence(args))

        if self.stage == 'syntactic':
            return
        self._set_dtype()
        self._set_shape_rank()

    @property
    def precedence(self):
        return self._precedence

    def _handle_precedence(self, args):
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

class PyccelUnaryOperator(PyccelOperator):

    def _set_dtype(self):
        a = self._args[0]
        if self._dtype is None:
            self._dtype     = a.dtype
        if self._precision is None:
            self._precision = a.precision

    def _set_shape_rank(self):
        a = self._args[0]
        if self._rank is None:
            self._rank      = a.rank
        if self._shape is None:
            self._shape     = a.shape

class PyccelUnary(PyccelUnaryOperator):
    _precedence = 14
    def _handle_precedence(self, args):
        args = PyccelUnaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

class PyccelUnarySub(PyccelUnary):
    pass

class PyccelNot(PyccelUnaryOperator):
    _precedence = 6
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']

class PyccelInvert(PyccelUnaryOperator):
    _precedence = 14
    _dtype     = NativeInteger()

    def _set_dtype(self):
        a = self._args[0]
        if self._args[0].dtype not in (NativeInteger(), NativeBool()):
            raise TypeError('unsupported operand type(s): {}'.format(self))

        self._args      = (PythonInt(a) if a.dtype is NativeBool() else a,)

        precision = a.precision

class PyccelAssociativeParenthesis(PyccelUnaryOperator):
    _precedence = 18
    def _handle_precedence(self, args):
        return args

class PyccelBinaryOperator(PyccelOperator):

    def __init__(self, *args):
        PyccelOperator.__init__(self, *args)

        if self.stage == 'syntactic':
            return
        self._set_dtype()
        self._set_shape_rank()


    def _set_dtype(self):
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
        self._dtype = NativeString()

    def _handle_complex_type(self, complexes):
        self._dtype     = NativeComplex()
        self._precision = max(a.precision for a in complexes)

    def _handle_real_type(self, reals):
        self._dtype     = NativeReal()
        self._precision = max(a.precision for a in reals)

    def _handle_integer_type(self, integers):
        self._dtype     = NativeInteger()
        self._precision = max(a.precision for a in integers)

    def _set_shape_rank(self):
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
                shape = self._broadcast()

                self._shape = shape
                self._rank  = len(shape)
            else:
                self._rank  = max(a.rank for a in self._args)
                self._shape = [None]*self._rank

    def _broadcast(self):
        shape = broadcast(self._args[0].shape, self._args[1].shape)

        for a in self._args[2:]:
            shape = broadcast(shape, a.shape)
        return shape

class PyccelArithmeticOperator(PyccelBinaryOperator):
    def _handle_precedence(self, args):
        args = PyccelBinaryOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelUnary) else a for a in args)
        return args

class PyccelPow(PyccelArithmeticOperator):
    _precedence  = 15

class PyccelAdd(PyccelArithmeticOperator):
    _precedence = 12

class PyccelMul(PyccelArithmeticOperator):
    _precedence = 13

class PyccelMinus(PyccelAdd):
    pass

class PyccelDiv(PyccelArithmeticOperator):
    _precedence = 13

    def _handle_str_type(self, strs):
        raise TypeError("unsupported operand type(s) for /: 'str' and 'str'")

    def _handle_integer_type(self, integers):
        self._dtype     = NativeReal()
        self._precision = default_precision['real']

class PyccelMod(PyccelArithmeticOperator):
    _precedence = 13

class PyccelFloorDiv(PyccelArithmeticOperator):
    _precedence = 13

class PyccelBitOperator(PyccelBinaryOperator):
    _rank = 0
    _shape = ()

    def _set_shape_rank(self):
        pass

    def _handle_str_type(self, strs):
        raise TypeError('unsupported operand type(s): {}'.format(self))

    def _handle_complex_type(self, complexes):
        raise TypeError('unsupported operand type(s): {}'.format(self))

    def _handle_real_type(self, reals):
        raise TypeError('unsupported operand type(s): {}'.format(self))

    def _handle_integer_type(self, integers):
        self._dtype     = NativeInteger()
        self._precision = max(a.precision for a in integers)
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in integers]

class PyccelRShift(PyccelBitOperator):
    _precedence = 11

class PyccelLShift(PyccelBitOperator):
    _precedence = 11

class PyccelBitComparisonOperator(PyccelBitOperator):
    def _handle_integer_type(self, integers):
        if all(a.dtype is NativeInteger() for a in integers):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in integers):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in integers]
        self._precision = max(a.precision for a in integers)

class PyccelBitXor(PyccelBitComparisonOperator):
    _precedence = 9

class PyccelBitOr(PyccelBitComparisonOperator):
    _precedence = 8

class PyccelBitAnd(PyccelBitComparisonOperator):
    _precedence = 10

class PyccelComparisonOperator(PyccelBinaryOperator):
    _precedence = 7
    _dtype = NativeBool()
    _precision = default_precision['bool']
    def _set_dtype(self):
        pass

class PyccelEq(PyccelComparisonOperator):
    pass
class PyccelNe(PyccelComparisonOperator):
    pass
class PyccelLt(PyccelComparisonOperator):
    pass
class PyccelLe(PyccelComparisonOperator):
    pass
class PyccelGt(PyccelComparisonOperator):
    pass
class PyccelGe(PyccelComparisonOperator):
    pass

class PyccelBooleanOperator(PyccelOperator):
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']

    def _set_dtype(self):
        pass
    def _set_shape_rank(self):
        pass

class PyccelAnd(PyccelBooleanOperator):
    _precedence = 5
    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelOr) else a for a in args)
        return args

class PyccelOr(PyccelBooleanOperator):
    _precedence = 4
    def _handle_precedence(self, args):
        args = PyccelBooleanOperator._handle_precedence(self, args)
        args = tuple(PyccelAssociativeParenthesis(a) if isinstance(a, PyccelAnd) else a for a in args)
        return args

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

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


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


Relational = (PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe, PyccelAnd, PyccelOr,  PyccelNot, Is, IsNot)

