from sympy.core.expr          import Expr
from .basic     import PyccelAstNode

from .builtins import PythonInt

from .core      import PyccelArraySize

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
        self._args = self._handle_precedence(args)

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
    def __init__(self, a):
        PyccelOperator.__init__(self, a)
        if self.stage == 'syntactic':
            return
        self._set_AST_node_values(self._args[0])

    def _set_AST_node_values(self, a):
        if self._dtype is None:
            self._dtype     = a.dtype
        if self._rank is None:
            self._rank      = a.rank
        if self._precision is None:
            self._precision = a.precision
        if self._shape is None:
            self._shape     = a.shape

class PyccelUnary(PyccelOperator):
    _precedence = 14

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

    def __init__(self, a):
        PyccelUnaryOperator.__init__(self,a)
        if self.stage == 'syntactic':
            return

        if self._args[0].dtype not in (NativeInteger(), NativeBool()):
            raise TypeError('unsupported operand type(s): {}'.format(self))

        self._args      = (PythonInt(a) if a.dtype is NativeBool() else a,)

class PyccelAssociativeParenthesis(PyccelOperator):
    _precedence = 18
    def _handle_precedence(self, args):
        pass

class PyccelBitOperator(PyccelOperator):
    _rank = 0
    _shape = ()

    def __init__(self, args):
        if self.stage == 'syntactic':
            return

        max_precision = 0
        for a in args:
            if a.dtype is NativeInteger() or a.dtype is NativeBool():
                max_precision = max(a.precision, max_precision)
            else:
                raise TypeError('unsupported operand type(s): {}'.format(self))
        self._precision = max_precision

class PyccelRShift(PyccelBitOperator):
    _precedence = 11
    _dtype = NativeInteger()
    def __init__(self, *args):
        super(PyccelRShift, self).__init__(args)
        if self.stage == 'syntactic':
            return
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelLShift(PyccelBitOperator):
    _precedence = 11
    _dtype = NativeInteger()
    def __init__(self, *args):
        super(PyccelLShift, self).__init__(args)
        if self.stage == 'syntactic':
            return
        self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBitXor(PyccelBitOperator):
    _precedence = 9
    def __init__(self, *args):
        super(PyccelBitXor, self).__init__(args)
        if self.stage == 'syntactic':
            return
        if all(a.dtype is NativeInteger() for a in args):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBitOr(PyccelBitOperator):
    _precedence = 8
    def __init__(self, *args):
        super(PyccelBitOr, self).__init__(args)
        if self.stage == 'syntactic':
            return
        if all(a.dtype is NativeInteger() for a in args):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBitAnd(PyccelBitOperator):
    _precedence = 10
    def __init__(self, *args):
        super(PyccelBitAnd, self).__init__(args)
        if self.stage == 'syntactic':
            return
        if all(a.dtype is NativeInteger() for a in args):
            self._dtype = NativeInteger()
        elif all(a.dtype is NativeBool() for a in args):
            self._dtype = NativeBool()
        else:
            self._dtype = NativeInteger()
            self._args = [PythonInt(a) if a.dtype is NativeBool() else a for a in args]

class PyccelBinaryOperator(PyccelOperator):

    def __init__(self, *args):

        PyccelOperator.__init__(self, *args)
        if self.stage == 'syntactic':
            return
        integers  = [a for a in args if a.dtype is NativeInteger() or a.dtype is NativeBool()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        strs      = [a for a in args if a.dtype is NativeString()]

        if strs:
            self._dtype = NativeString()
            self._rank  = 0
            self._shape = ()
            assert len(integers + reals + complexes) == 0
        else:
            if complexes:
                self._dtype     = NativeComplex()
                self._precision = max(a.precision for a in complexes)
            elif reals:
                self._dtype     = NativeReal()
                self._precision = max(a.precision for a in reals)
            elif integers:
                self._dtype     = NativeInteger()
                self._precision = max(a.precision for a in integers)
            else:
                raise TypeError('cannot determine the type of {}'.format(self))

            ranks  = [a.rank for a in args]
            shapes = [a.shape for a in args]

            if None in ranks:
                self._rank  = None
                self._shape = None
            elif all(sh is not None for tup in shapes for sh in tup):
                if len(args) == 1:
                    shape = args[0].shape
                else:
                    shape = broadcast(args[0].shape, args[1].shape)

                    for a in args[2:]:
                        shape = broadcast(shape, a.shape)

                self._shape = shape
                self._rank  = len(shape)
            else:
                self._rank  = max(a.rank for a in args)
                self._shape = [None]*self._rank

class PyccelPow(PyccelBinaryOperator):
    _precedence  = 15
class PyccelAdd(PyccelBinaryOperator):
    _precedence = 12
class PyccelMul(PyccelBinaryOperator):
    _precedence = 13
class PyccelMinus(PyccelAdd):
    pass
class PyccelDiv(PyccelBinaryOperator):
    _precedence = 13
    def __init__(self, *args):

        PyccelOperator.__init__(self, *args)
        if self.stage == 'syntactic':
            return

        integers  = [a for a in args if a.dtype is NativeInteger() or a.dtype is NativeBool()]
        reals     = [a for a in args if a.dtype is NativeReal()]
        complexes = [a for a in args if a.dtype is NativeComplex()]
        if complexes:
            self._dtype     = NativeComplex()
            self._precision = max(a.precision for a in complexes)
        elif reals:
            self._dtype     = NativeReal()
            self._precision = max(a.precision for a in reals)
        elif integers:
            self._dtype     = NativeReal()
            self._precision = default_precision['real']

        ranks  = [a.rank for a in args]
        shapes = [a.shape for a in args]

        if None in ranks:
            self._rank  = None
            self._shape = None

        elif all(sh is not None for tup in shapes for sh in tup):
            shape = broadcast(args[0].shape, args[1].shape)

            for a in args[2:]:
                shape = broadcast(shape, a.shape)

            self._shape = shape
            self._rank  = len(shape)
        else:
            self._rank  = max(a.rank for a in args)
            self._shape = [None]*self._rank

class PyccelMod(PyccelBinaryOperator):
    _precedence = 13
class PyccelFloorDiv(PyccelBinaryOperator):
    _precedence = 13

class PyccelBooleanOperator(PyccelOperator):
    _precedence = 7

    def __init__(self, *args):

        PyccelOperator.__init__(self, *args)
        if self.stage == 'syntactic':
            return

        self._dtype = NativeBool()
        self._precision = default_precision['bool']

        ranks  = [a.rank for a in args]
        shapes = [a.shape for a in args]

        if None in ranks:
            self._rank  = None
            self._shape = None

        elif all(sh is not None for tup in shapes for sh in tup):
            shape = broadcast(args[0].shape, args[1].shape)
            for a in args[2:]:
                shape = broadcast(shape, a.shape)

            self._shape = shape
            self._rank  = len(shape)
        else:
            self._rank = max(a.rank for a in args)
            self._shape = [None]*self._rank

class PyccelEq(PyccelBooleanOperator):
    pass
class PyccelNe(PyccelBooleanOperator):
    pass
class PyccelLt(PyccelBooleanOperator):
    pass
class PyccelLe(PyccelBooleanOperator):
    pass
class PyccelGt(PyccelBooleanOperator):
    pass
class PyccelGe(PyccelBooleanOperator):
    pass

class PyccelAnd(PyccelOperator):
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 5

    def __init__(self, *args):
        PyccelOperator.__init__(self, *args)

class PyccelOr(PyccelOperator):
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 4

    def __init__(self, *args):
        PyccelOperator.__init__(self, *args)

class Is(PyccelOperator):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast import Is
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> Is(x, Nil())
    Is(x, None)
    """
    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 7

    def __init__(self, *args):
        PyccelOperator.__init__(self, *args)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


class IsNot(PyccelOperator):

    """Represents a is expression in the code.

    Examples
    --------
    >>> from pyccel.ast import IsNot
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> IsNot(x, Nil())
    IsNot(x, None)
    """

    _dtype = NativeBool()
    _rank  = 0
    _shape = ()
    _precision = default_precision['bool']
    _precedence = 7

    def __init__(self, *args):
        PyccelOperator.__init__(self, *args)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


Relational = (PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe, PyccelAnd, PyccelOr,  PyccelNot, Is, IsNot)

