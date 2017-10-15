# coding: utf-8

from __future__ import print_function, division

from numpy import ndarray

from sympy.core.expr import Expr
from sympy.core import Symbol, Tuple
from sympy.core.relational import Equality, Relational,Ne,Eq
from sympy.logic.boolalg import And, Boolean, Not, Or, true, false
from sympy.core.singleton import Singleton
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy import sympify
from sympy import Symbol, Integer, Add, Mul,Pow
from sympy import Float as Sympy_Float
from sympy.core.compatibility import with_metaclass
from sympy.core.compatibility import is_sequence
from sympy.sets.fancysets import Range as sm_Range
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
from sympy.utilities.iterables import iterable
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse

from sympy.core.basic import Basic
from sympy.core.expr import Expr, AtomicExpr
from sympy.core.compatibility import string_types
from sympy.core.operations import LatticeOp
from sympy.core.function import Derivative
from sympy.core.function import _coeff_isneg
from sympy.core.singleton import S
from sympy.utilities.iterables import iterable
from sympy import Integral, Symbol
from sympy.simplify.radsimp import fraction
from sympy.logic.boolalg import BooleanFunction

import collections
from sympy.core.compatibility import is_sequence

# TODO: add examples: Break, Len, Shape,
#                     Min, Max, Dot, Sign, Array,
#                     Thread, ThreadID, ThreadNumber
# TODO: add EmptyStmt => empty lines
# TODO: clean Thread objects
# TODO: update code examples

# TODO add examples
# TODO treat Function case
# TODO treat Zeros, Ones, Array cases
# TODO treat AnnotatedComment case
# TODO treat Slice case
# TODO treat Thread cases
# TODO treat Stencil case
# TODO treat FunctionHeader case
def subs(expr, a_old, a_new):
    """
    Substitutes old for new in an expression after sympifying args.

    a_old: str, Symbol, Variable
        name of the symbol to replace
    a_new: str, Symbol, Variable
        name of the new symbol

    Examples
    """
    a_new = a_old.clone(str(a_new))

    if iterable(expr):
        return [subs(i, a_old, a_new) for i in expr]
    elif isinstance(expr, Variable):
        if expr.name == str(a_old):
            return a_new
        else:
            return expr
    elif isinstance(expr, IndexedVariable):
        if str(expr) == str(a_old):
            return IndexedVariable(str(a_new))
        else:
            return expr
    elif isinstance(expr, IndexedElement):
        base    = subs(expr.base   , a_old, a_new)
        indices = subs(expr.indices, a_old, a_new)
        return base[indices]
    elif isinstance(expr, Expr):
        return expr.subs({a_old: a_new})
    elif isinstance(expr, Zeros):
        e_lhs   = subs(expr.lhs, a_old, a_new)
        e_shape = subs(expr.shape, a_old, a_new)
        return Zeros(e_lhs, e_shape)
    elif isinstance(expr, Ones):
        e_lhs   = subs(expr.lhs, a_old, a_new)
        e_shape = subs(expr.shape, a_old, a_new)
        return Ones(e_lhs, e_shape)
    elif isinstance(expr, ZerosLike):
        e_rhs = subs(expr.rhs, a_old, a_new)
        e_lhs = subs(expr.lhs, a_old, a_new)
        return ZerosLike(e_lhs, e_rhs)
    elif isinstance(expr, Assign):
        e_rhs = subs(expr.rhs, a_old, a_new)
        e_lhs = subs(expr.lhs, a_old, a_new)
        return Assign(e_lhs, e_rhs, strict=False)
    elif isinstance(expr, MultiAssign):
        e_rhs   = subs(expr.rhs, a_old, a_new)
        e_lhs   = subs(expr.lhs, a_old, a_new)
        return MultiAssign(e_lhs, e_rhs)
    elif isinstance(expr, While):
        test = subs(expr.test, a_old, a_new)
        body = subs(expr.body, a_old, a_new)
        return While(test, body)
    elif isinstance(expr, For):
        # TODO treat iter correctly
#        target   = subs(expr.target, a_old, a_new)
#        it       = subs(expr.iterable, a_old, a_new)
        target   = expr.target
        it       = expr.iterable
        body     = subs(expr.body, a_old, a_new)
        return For(target, it, body)
    elif isinstance(expr, If):
        args = []
        for block in expr.args:
            test  = block[0]
            stmts = block[1]
            t = subs(test,  a_old, a_new)
            s = subs(stmts, a_old, a_new)
            args.append((t,s))
        return If(*args)
    elif isinstance(expr, FunctionDef):
        name        = subs(expr.name, a_old, a_new)
        arguments   = subs(expr.arguments, a_old, a_new)
        results     = subs(expr.results, a_old, a_new)
        body        = subs(expr.body, a_old, a_new)
        local_vars  = subs(expr.local_vars, a_old, a_new)
        global_vars = subs(expr.global_vars, a_old, a_new)
        return FunctionDef(name, arguments, results, \
                           body, local_vars, global_vars)
    elif isinstance(expr, Declare):
        dtype     = subs(expr.dtype, a_old, a_new)
        variables = subs(expr.variables, a_old, a_new)
        return Declare(dtype, variables)
    else:
        return expr

def allocatable_like(expr, verbose=False):
    """
    finds attributs of an expression

    expr: Expr
        a pyccel expression

    verbose: bool
        talk more
    """
#    print ('>>>>> expr = ', expr)
#    print ('>>>>> type = ', type(expr))

    if isinstance(expr, (Variable, IndexedVariable, IndexedElement)):
        return expr
    elif isinstance(expr, Expr):
        args = [expr]
        while args:
            a = args.pop()
#            print (">>>> ", a, type(a))

            # XXX: This is a hack to support non-Basic args
            if isinstance(a, string_types):
                continue

            if a.is_Mul:
                if _coeff_isneg(a):
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                    else:
                        args.append(ai)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if (a.is_Mul or
                a.is_Pow or
                a.is_Function or
                isinstance(a, Derivative) or
                    isinstance(a, Integral)):

                o = Symbol(a.func.__name__.upper())
            if     (not a.is_Symbol) \
               and (not isinstance(a, (IndexedElement, Function))):
                args.extend(a.args)
            if isinstance(a, Function):
                if verbose:
                    print ("Functions not yet available")
                return None
            elif isinstance(a, (Variable, IndexedVariable, IndexedElement)):
                return a
            elif a.is_Symbol:
                raise TypeError("Found an unknown symbol {0}".format(str(a)))
    else:
        raise TypeError("Unexpected type")

class DottedName(Basic):
    """
    Represents a dotted variable.

    Examples

    >>> from pyccel.types.ast import DottedName
    >>> DottedName('matrix', 'n_rows')
    matrix.n_rows
    >>> DottedName('pyccel', 'mpi', 'mpi_init')
    pyccel.mpi.mpi_init
    """
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

    @property
    def name(self):
        return self._args

    def __str__(self):
        return '.'.join(str(n) for n in self.name)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '.'.join(sstr(n) for n in self.name)

class Assign(Basic):
    """Represents variable assignment for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    strict: bool
        if True, we do some verifications. In general, this can be more
        complicated and is treated in pyccel.syntax.

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from pyccel.types.ast import Assign
    >>> x, y, z = symbols('x, y, z')
    >>> Assign(x, y)
    x := y
    >>> Assign(x, 0)
    x := 0
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assign(A, mat)
    A := Matrix([[x, y, z]])
    >>> Assign(A[0, 1], x)
    A[0, 1] := x

    """

    def __new__(cls, lhs, rhs, strict=False, status=None, like=None):
        cls._strict = strict
        if strict:
            lhs = sympify(lhs)
            rhs = sympify(rhs)
            # Tuple of things that can be on the lhs of an assignment
            assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
            #if not isinstance(lhs, assignable):
            #    raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
            # Indexed types implement shape, but don't define it until later. This
            # causes issues in assignment validation. For now, matrices are defined
            # as anything with a shape that is not an Indexed
            lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
            rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)
            # If lhs and rhs have same structure, then this assignment is ok
            if lhs_is_mat:
                if not rhs_is_mat:
                    raise ValueError("Cannot assign a scalar to a matrix.")
                elif lhs.shape != rhs.shape:
                    raise ValueError("Dimensions of lhs and rhs don't align.")
            elif rhs_is_mat and not lhs_is_mat:
                raise ValueError("Cannot assign a matrix to a scalar.")
        return Basic.__new__(cls, lhs, rhs, status, like)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    # TODO : remove
    @property
    def expr(self):
        return self.rhs

    @property
    def status(self):
        return self._args[2]

    @property
    def like(self):
        return self._args[3]

    @property
    def strict(self):
        return self._strict


# The following are defined to be sympy approved nodes. If there is something
# smaller that could be used, that would be preferable. We only use them as
# tokens.


class NativeOp(with_metaclass(Singleton, Basic)):
    """Base type for native operands."""
    pass


class AddOp(NativeOp):
    _symbol = '+'


class SubOp(NativeOp):
    _symbol = '-'


class MulOp(NativeOp):
    _symbol = '*'


class DivOp(NativeOp):
    _symbol = '/'


class ModOp(NativeOp):
    _symbol = '%'


op_registry = {'+': AddOp(),
               '-': SubOp(),
               '*': MulOp(),
               '/': DivOp(),
               '%': ModOp()}


def operator(op):
    """Returns the operator singleton for the given operator"""

    if op.lower() not in op_registry:
        raise ValueError("Unrecognized operator " + op)
    return op_registry[op]


class AugAssign(Basic):
    """
    Represents augmented variable assignment for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : NativeOp
        Operator (+, -, /, \*, %).

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    strict: bool
        if True, we do some verifications. In general, this can be more
        complicated and is treated in pyccel.syntax.

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.types.ast import AugAssign
    >>> s = Variable('int', 's')
    >>> t = Variable('int', 't')
    >>> AugAssign(s, '+', 2 * t + 1)
    s += 1 + 2*t
    """

    def __new__(cls, lhs, op, rhs, strict=False, status=None, like=None):
        cls._strict = strict
        if strict:
            lhs = sympify(lhs)
            rhs = sympify(rhs)
            # Tuple of things that can be on the lhs of an assignment
            assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed)
            if not isinstance(lhs, assignable):
                raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
            # Indexed types implement shape, but don't define it until later. This
            # causes issues in assignment validation. For now, matrices are defined
            # as anything with a shape that is not an Indexed
            lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
            rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)
            # If lhs and rhs have same structure, then this assignment is ok
            if lhs_is_mat:
                if not rhs_is_mat:
                    raise ValueError("Cannot assign a scalar to a matrix.")
                elif lhs.shape != rhs.shape:
                    raise ValueError("Dimensions of lhs and rhs don't align.")
            elif rhs_is_mat and not lhs_is_mat:
                raise ValueError("Cannot assign a matrix to a scalar.")

        if isinstance(op, str):
            op = operator(op)
        elif op not in op_registry.values():
            raise TypeError("Unrecognized Operator")

        return Basic.__new__(cls, lhs, op, rhs, status, like)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} {1}= {2}'.format(sstr(self.lhs), self.op._symbol,
                sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def op(self):
        return self._args[1]

    @property
    def rhs(self):
        return self._args[2]

    @property
    def status(self):
        return self._args[3]

    @property
    def like(self):
        return self._args[4]

    @property
    def strict(self):
        return self._strict

class While(Basic):
    """Represents a 'while' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    test : expression
        test condition given as a sympy expression
    body : sympy expr
        list of statements representing the body of the While statement.

    Examples

    >>> from sympy import Symbol
    >>> from pyccel.types.ast import Assign, While
    >>> n = Symbol('n')
    >>> While((n>1), [Assign(n,n-1)])
    While(n > 1, (n := n - 1,))
    """
    def __new__(cls, test, body):
        test = sympify(test)

        if not iterable(body):
            raise TypeError("body must be an iterable")
        body = Tuple(*(sympify(i) for i in body))
        return Basic.__new__(cls, test, body)

    @property
    def test(self):
        return self._args[0]


    @property
    def body(self):
        return self._args[1]

class Range(sm_Range):
    """
    Representes a range.

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.types.ast import Range
    >>> from sympy import Symbol
    >>> s = Variable('int', 's')
    >>> e = Symbol('e')
    >>> Range(s, e, 1)
    Range(0, n, 1)
    """

    def __new__(cls, *args):
        _args = [1, 1, 1]
        r = sm_Range.__new__(cls, *_args)
        r._args = args

        return r

    @property
    def start(self):
        return self._args[0]

    @property
    def stop(self):
        return self._args[1]

    @property
    def step(self):
        return self._args[2]

    @property
    def size(self):
        return (self.stop - self.start)/self.step

class Tile(Range):
    """
    Representes a tile.

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.types.ast import Tile
    >>> from sympy import Symbol
    >>> s = Variable('int', 's')
    >>> e = Symbol('e')
    >>> Tile(s, e, 1)
    Tile(0, n, 1)
    """

    def __new__(cls, start, stop):
        step = 1
        return Range.__new__(cls, start, stop, step)

    @property
    def start(self):
        return self._args[0]

    @property
    def stop(self):
        return self._args[1]

    @property
    def size(self):
        return self.stop - self.start

# TODO: implement it as an extension of sympy Tensor?
class Tensor(Basic):
    """
    Base class for tensor.

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.types.ast import Range, Tensor
    >>> from sympy import Symbol
    >>> s1 = Variable('int', 's1')
    >>> s2 = Variable('int', 's2')
    >>> e1 = Variable('int', 'e1')
    >>> e2 = Variable('int', 'e2')
    >>> r1 = Range(s1, e1, 1)
    >>> r2 = Range(s2, e2, 1)
    >>> Tensor(r1, r2)
    Tensor(Range(s1, e1, 1), Range(s2, e2, 1), name=tensor)
    """

    def __new__(cls, *args, **kwargs):
        for r in args:
            if not isinstance(r, (Range, Tensor)):
                raise TypeError("Expecting a Range or Tensor")

        try:
            name = kwargs['name']
        except:
            name = 'tensor'

        args = list(args) + [name]

        return Basic.__new__(cls, *args)

    @property
    def name(self):
        return self._args[-1]

    @property
    def ranges(self):
        return self._args[:-1]

    @property
    def dim(self):
        return len(self.ranges)

    def _sympystr(self, printer):
        sstr = printer.doprint
        txt  = ', '.join(sstr(n) for n in self.ranges)
        txt  = 'Tensor({0}, name={1})'.format(txt, sstr(self.name))
        return txt

# TODO add a name to a block?
class Block(Basic):
    """Represents a block in the code. A block consists of the following inputs

    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    Examples

    >>> from pyccel.types.ast import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> Block([n, x], [Assign(x,2.*n + 1.), Assign(n, n + 1)])
    Block([n, x], [x := 1.0 + 2.0*n, n := 1 + n])
    """

    def __new__(cls, variables, body):
        if not iterable(variables):
            raise TypeError("variables must be an iterable")
        for var in variables:
            if not isinstance(var, Variable):
                raise TypeError("Only a Variable instance is allowed.")
        if not iterable(body):
            raise TypeError("body must be an iterable")
        return Basic.__new__(cls, variables, body)

    @property
    def variables(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]

class Module(Basic):
    """Represents a block in the code. A block consists of the following inputs

    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    funcs: list
        a list of FunctionDef instances

    classes: list
        a list of ClassDef instances

    Examples

    >>> from pyccel.types.ast import Variable, Assign
    >>> from pyccel.types.ast import ClassDef, FunctionDef, Module
    >>> x = Variable('double', 'x')
    >>> y = Variable('double', 'y')
    >>> z = Variable('double', 'z')
    >>> t = Variable('double', 't')
    >>> a = Variable('double', 'a')
    >>> b = Variable('double', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributs   = [x,y]
    >>> methods     = [translate]
    >>> Point = ClassDef('Point', attributs, methods)
    >>> incr = FunctionDef('incr', [x], [y], [Assign(y,x+1)])
    >>> decr = FunctionDef('decr', [x], [y], [Assign(y,x-1)])
    >>> Module('my_module', [], [incr, decr], [Point])
    Module(my_module, [], [FunctionDef(incr, (x,), (y,), [y := 1 + x], [], [], None, False, function), FunctionDef(decr, (x,), (y,), [y := -1 + x], [], [], None, False, function)], [ClassDef(Point, (x, y), (FunctionDef(translate, (x, y, a, b), (z, t), [y := a + x], [], [], None, False, function),), [public])])
    """

    def __new__(cls, name, variables, funcs, classes):
        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError("variables must be an iterable")
        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError("Only a Variable instance is allowed.")

        if not iterable(funcs):
            raise TypeError("funcs must be an iterable")
        for i in funcs:
            if not isinstance(i, FunctionDef):
                raise TypeError("Only a FunctionDef instance is allowed.")

        if not iterable(classes):
            raise TypeError("classes must be an iterable")
        for i in classes:
            if not isinstance(i, ClassDef):
                raise TypeError("Only a ClassDef instance is allowed.")

        return Basic.__new__(cls, name, variables, funcs, classes)

    @property
    def name(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

    @property
    def funcs(self):
        return self._args[2]

    @property
    def classes(self):
        return self._args[3]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]

    @property
    def body(self):
        return self.funcs + self.classes

class For(Basic):
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    target : symbol
        symbol representing the iterator
    iter : iterable
        iterable object. for the moment only Range is used
    body : sympy expr
        list of statements representing the body of the For statement.

    Examples

    >>> from sympy import symbols, MatrixSymbol
    >>> from pyccel.types.ast import Assign, For
    >>> i,b,e,s,x = symbols('i,b,e,s,x')
    >>> A = MatrixSymbol('A', 1, 3)
    >>> For(i, (b,e,s), [Assign(x,x-1), Assign(A[0, 1], x)])
    For(i, Range(b, e, s), (x := x - 1, A[0, 1] := x))
    """

    def __new__(cls, target, iter, body, strict=True):
        if strict:
            target = sympify(target)
            if not iterable(iter) and not isinstance(iter, (Range, Tensor)):
                raise TypeError("iter must be an iterable")
            if not isinstance(iter, (Range, Tensor)):
                raise TypeError("Expecting a Range or Tensor")
            if not iterable(body):
                raise TypeError("body must be an iterable")
            body = Tuple(*(sympify(i) for i in body))
        return Basic.__new__(cls, target, iter, body)

    @property
    def target(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]


# The following are defined to be sympy approved nodes. If there is something
# smaller that could be used, that would be preferable. We only use them as
# tokens.


class DataType(with_metaclass(Singleton, Basic)):
    """Base class representing native datatypes"""
    _name = '__UNDEFINED__'

    @property
    def name(self):
        return self._name

class NativeBool(DataType):
    _name = 'Bool'
    pass

class NativeInteger(DataType):
    _name = 'Int'
    pass

class NativeFloat(DataType):
    _name = 'Float'
    pass

class NativeDouble(DataType):
    _name = 'Double'
    pass

class NativeComplex(DataType):
    _name = 'Complex'
    pass

class NativeVoid(DataType):
    _name = 'Void'
    pass

class NativeRange(DataType):
    _name = 'Range'
    pass

class NativeTensor(DataType):
    _name = 'Tensor'
    pass

class CustomDataType(DataType):
    _name = '__UNDEFINED__'
    pass


Bool = NativeBool()
Int = NativeInteger()
Float = NativeFloat()
Double = NativeDouble()
Complex = NativeComplex()
Void = NativeVoid()


dtype_registry = {'bool': Bool,
                  'int': Int,
                  'float': Float,
                  'double': Double,
                  'complex': Complex,
                  'void': Void}


def DataTypeFactory(name, argnames=["_name"], \
                    BaseClass=CustomDataType, \
                    prefix='Pyccel', \
                    alias=None):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            # here, the argnames variable is the one passed to the
            # DataTypeFactory call
            if key not in argnames:
                raise TypeError("Argument %s not valid for %s"
                    % (key, self.__class__.__name__))
            setattr(self, key, value)
        BaseClass.__init__(self, name[:-len("Class")])

    newclass = type(prefix + name, (BaseClass,), \
                    {"__init__": __init__, \
                     "prefix":   prefix, \
                     "alias":    alias})
    return newclass

def is_pyccel_datatype(expr):
    return isinstance(expr, CustomDataType)
#    if not isinstance(expr, DataType):
#        raise TypeError('Expecting a DataType instance')
#    name = expr.__class__.__name__
#    return name.startswith('Pyccel')

# TODO check the use of floats
def datatype(arg):
    """Returns the datatype singleton for the given dtype.

    arg : str or sympy expression
        If a str ('bool', 'int', 'float', 'double', or 'void'), return the
        singleton for the corresponding dtype. If a sympy expression, return
        the datatype that best fits the expression. This is determined from the
        assumption system. For more control, use the `DataType` class directly.

    Returns:
        DataType

    """
    def infer_dtype(arg):
        if arg.is_integer:
            return Int
        elif arg.is_Boolean:
            return Bool
        else:
            return Double

    if isinstance(arg, str):
        if arg.lower() not in dtype_registry:
            raise ValueError("Unrecognized datatype " + arg)
        return dtype_registry[arg]
    else:
        arg = sympify(arg)
        if isinstance(arg, ImmutableDenseMatrix):
            dts = [infer_dtype(i) for i in arg]
            if all([i is Bool for i in dts]):
                return Bool
            elif all([i is Int for i in dts]):
                return Int
            else:
                return Double
        else:
            return infer_dtype(arg)

class EqualityStmt(Relational):
    """Represents a relational equality expression in the code."""
    def __new__(cls,lhs,rhs):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)
    @property
    def canonical(self):
        return self

class NotequalStmt(Relational):
    """Represents a relational not equality expression in the code."""
    def __new__(cls,lhs,rhs):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class FunctionCall(Basic):
    """
    Base class for applied mathematical functions.

    It also serves as a constructor for undefined function classes.

    func: FunctionDef, str
        an instance of FunctionDef or function name

    arguments: list, tuple, None
        a list of arguments.

    kind: str
        'function' or 'procedure'. default value: 'function'

    Examples

    """

    def __new__(cls, func, arguments, kind='function'):
        if not isinstance(func, (FunctionDef, str)):
            raise TypeError("Expecting func to be a FunctionDef or str")

        if isinstance(func, FunctionDef):
            kind = func.kind

        if not isinstance(kind, str):
            raise TypeError("Expecting a string for kind.")

        if not (kind in ['function', 'procedure']):
            raise ValueError("kind must be one among {'function', 'procedure'}")

        return Basic.__new__(cls, func, arguments, kind)

    def _sympystr(self, printer):
        sstr = printer.doprint
        name = sstr(self.name)
        args = ''
        if not(self.arguments) is None:
            args = ', '.join(sstr(i) for i in self.arguments)
        return '{0}({1})'.format(name, args)

    @property
    def func(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def kind(self):
        return self._args[2]

    @property
    def name(self):
        if isinstance(self.func, FunctionDef):
            return self.func.name
        else:
            return self.func

class Variable(Symbol):
    """Represents a typed variable.

    dtype : str, DataType
        The type of the variable. Can be either a DataType,
        or a str (bool, int, float, double).

    name : str, list
        The sympy object the variable represents. This can be either a string
        or a dotted name, when using a Class attribut.

    rank : int
        used for arrays. [Default value: 0]

    allocatable: False
        used for arrays, if we need to allocate memory [Default value: False]

    shape: int or list
        shape of the array. [Default value: None]

    cls_base: class
        class base if variable is an object or an object member

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Variable
    >>> x, n = symbols('x, n')
    >>> Variable('int', 'n')
    n
    >>> Variable('float', x, rank=2, shape=(n,2), allocatable=True)
    x
    >>> Variable('int', ('matrix', 'n_rows'))
    matrix.n_rows
    """
    def __new__(cls, dtype, name, \
                rank=0, allocatable=False, \
                shape=None, cls_base=None):

        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")

        # if class attribut
        if isinstance(name, str):
            name = name.split('.')
            if len(name) == 1:
                name = name[0]
            else:
                name = DottedName(*name)

        if not isinstance(name, (str, DottedName)):
            raise TypeError("Expecting a string or DottedName.")

        if not isinstance(rank, int):
            raise TypeError("rank must be an instance of int.")
#        if not shape==None:
#            if  (not isinstance(shape,int) and not isinstance(shape,tuple) and not all(isinstance(n, int) for n in shape)):
#                raise TypeError("shape must be an instance of int or tuple of int")

        return Basic.__new__(cls, dtype, name, rank, allocatable, shape, cls_base)

    @property
    def dtype(self):
        return self._args[0]

    @property
    def name(self):
        return self._args[1]

    @property
    def rank(self):
        return self._args[2]

    @property
    def allocatable(self):
        return self._args[3]

    @property
    def shape(self):
        return self._args[4]

    @property
    def cls_base(self):
        return self._args[5]

    def __str__(self):
        if isinstance(self.name, (str, DottedName)):
            return str(self.name)
        else:
            return '.'.join(print(n) for n in self.name)

    def _sympystr(self, printer):
        sstr = printer.doprint
        if isinstance(self.name, (str, DottedName)):
            return '{}'.format(sstr(self.name))
        else:
            return '.'.join(sstr(n) for n in self.name)

    def clone(self, name):
        cls = eval(self.__class__.__name__)
        return cls(self.dtype, name, \
                   rank=self.rank, \
                   allocatable=self.allocatable, \
                   shape=self.shape)

class FunctionDef(Basic):
    """Represents a function definition.

    name : str
        The name of the function.

    arguments : iterable
        The arguments to the function.

    results : iterable
        The direct outputs of the function.

    body : iterable
        The body of the function.

    local_vars : list of Symbols
        These are used internally by the routine.

    global_vars : list of Symbols
        Variables which will not be passed into the function.

    cls_name: str
        Class name if the function is a method of cls_name

    hide: bool
        if True, the function definition will not be generated.

    kind: str
        'function' or 'procedure'. default value: 'function'

    >>> from pyccel.types.ast import Assign, Variable, FunctionDef
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> args        = [x]
    >>> results     = [y]
    >>> body        = [Assign(y,x+1)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x,), (y,), [y := 1 + x], [], [], None, False, function)
    """

    def __new__(cls, name, arguments, results, \
                body, local_vars=[], global_vars=[], \
                cls_name=None, hide=False, kind='function'):
        # name
        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, Symbol):
            raise TypeError("Function name must be Symbol or string")
        # arguments
        if not iterable(arguments):
            raise TypeError("arguments must be an iterable")
        # TODO improve and uncomment
#        if not all(isinstance(a, Argument) for a in arguments):
#            raise TypeError("All arguments must be of type Argument")
        arguments = Tuple(*arguments)
        # body
        if not iterable(body):
            raise TypeError("body must be an iterable")
#        body = Tuple(*(i for i in body))
        # results
        if not iterable(results):
            raise TypeError("results must be an iterable")
        # TODO improve and uncomment
#        if not all(isinstance(i, Result) for i in results):
#            raise TypeError("All results must be of type Result")
        results = Tuple(*results)
        # if method
        if cls_name:
            if not(isinstance(cls_name, str)):
                raise TypeError("cls_name must be a string")

        if not isinstance(kind, str):
            raise TypeError("Expecting a string for kind.")

        if not (kind in ['function', 'procedure']):
            raise ValueError("kind must be one among {'function', 'procedure'}")

        return Basic.__new__(cls, name, \
                             arguments, results, \
                             body, \
                             local_vars, global_vars, \
                             cls_name, hide, kind)

    @property
    def name(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def results(self):
        return self._args[2]

    @property
    def body(self):
        return self._args[3]

    @property
    def local_vars(self):
        return self._args[4]

    @property
    def global_vars(self):
        return self._args[5]

    @property
    def cls_name(self):
        return self._args[6]

    @property
    def hide(self):
        return self._args[7]

    @property
    def kind(self):
        return self._args[8]

    def print_body(self):
        for s in self.body:
            print (s)

#    @property
#    def declarations(self):
#        ls = self.arguments + self.results + self.local_vars
#        return [Declare(i.dtype, i) for i in ls]

    def rename(self, newname):
        """
        Rename the FunctionDef name by creating a new FunctionDef with
        newname.

        newname: str
            new name for the FunctionDef
        """
        return FunctionDef(newname, self.arguments, self.results, self.body, \
                           local_vars=self.local_vars, \
                           global_vars=self.global_vars, \
                           cls_name=self.cls_name, \
                           hide=self.hide, \
                           kind=self.kind)

class ClassDef(Basic):
    """Represents a class definition.

    name : str
        The name of the class.
    attributs: iterable
        The attributs to the class.
    methods: iterable
        Class methods
    options: list, tuple
        list of options ('public', 'private', 'abstract')

    Examples

    >>> from pyccel.types.ast import Variable, Assign
    >>> from pyccel.types.ast import ClassDef, FunctionDef
    >>> x = Variable('double', 'x')
    >>> y = Variable('double', 'y')
    >>> z = Variable('double', 'z')
    >>> t = Variable('double', 't')
    >>> a = Variable('double', 'a')
    >>> b = Variable('double', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributs   = [x,y]
    >>> methods     = [translate]
    >>> ClassDef('Point', attributs, methods)
    ClassDef(Point, (x, y), (FunctionDef(translate, (x, y, a, b), (z, t), [y := a + x], [], [], None, False, function),), [public])
    """

    def __new__(cls, name, attributs, methods, options=['public']):
        # name
        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, Symbol):
            raise TypeError("Function name must be Symbol or string")
        # attributs
        if not iterable(attributs):
            raise TypeError("attributs must be an iterable")
        attributs = Tuple(*attributs)
        # methods
        if not iterable(methods):
            raise TypeError("methods must be an iterable")
        methods = Tuple(*methods)
        # options
        if not iterable(options):
            raise TypeError("options must be an iterable")

        return Basic.__new__(cls, name, attributs, methods, options)

    @property
    def name(self):
        return self._args[0]

    @property
    def attributs(self):
        return self._args[1]

    @property
    def methods(self):
        return self._args[2]

    @property
    def options(self):
        return self._args[3]

class Ceil(Function):
    """
    Represents ceil expression in the code.

    rhs: symbol or number
        input for the ceil function

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Ceil, Variable
    >>> n,x,y = symbols('n,x,y')
    >>> var = Variable('float', x)
    >>> Ceil(x)
    Ceil(x)
    >>> Ceil(var)
    Ceil(x)
    """
    def __new__(cls,rhs):
        return Basic.__new__(cls,rhs)

    @property
    def rhs(self):
        return self._args[0]

class Import(Basic):
    """Represents inclusion of dependencies in the code.

    fil : str
        The filepath of the module (i.e. header in C).
    funcs
        The name of the function (or an iterable of names) to be imported.

    Examples

    >>> from pyccel.types.ast import Import
    >>> Import('numpy', 'linspace')
    Import(numpy, (linspace,))

    >>> from pyccel.types.ast import DottedName
    >>> from pyccel.types.ast import Import
    >>> mpi = DottedName('pyccel', 'mpi')
    >>> Import(mpi, 'mpi_init')
    Import(pyccel.mpi, (mpi_init,))
    >>> Import(mpi, '*')
    Import(pyccel.mpi, (*,))
    """

    def __new__(cls, fil, funcs=None):
        if not isinstance(fil, (str, DottedName)):
            raise TypeError('Expecting a string or DottedName')

        if iterable(funcs):
            funcs = Tuple(*[Symbol(f) for f in funcs])
        elif not isinstance(funcs, (str, DottedName)):
            raise TypeError("Unrecognized funcs type: ", funcs)

        return Basic.__new__(cls, fil, funcs)

    @property
    def fil(self):
        return self._args[0]

    @property
    def funcs(self):
        return self._args[1]



class Result(Basic):

    """Represents a list of return variables and there return value in a fcuntion in the code.

    result_variables: a list of tuples each tuple have the variable return
                    and it's return value if it's an expression
    Example:
    >>> from pyccel.types.ast import  Variable
    >>> Result([(Variable('int', 'n'),n*2]),(Variable('int', 'x'),None]))

    """
    def __new__(cls,result_variables):
        if isinstance(result_variables,list):
            for i in result_variables:
                if not isinstance(i[0],Variable):
                    raise TypeError("{0} must be of type Variable".format(i[0]))
                if not i[1]==None:
                    if not isinstance(i[1],(Integer,Sympy_Float,Add,Mul,Pow)):
                        raise TypeError("{0} must be a sympy Expression".format(i[1]))
        else:
            raise TypeError("result_variables must be a type list ")

        return Basic.__new__(cls,result_variables)

    @property
    def result_variables(self):
        return self._args[0]



# TODO: Should Declare have an optional init value for each var?


class Declare(Basic):
    """Represents a variable declaration in the code.

    dtype : DataType
        The type for the declaration.
    variable(s)
        A single variable or an iterable of Variables. If iterable, all
        Variables must be of the same type.
    intent: None, str
        one among {'in', 'out', 'inout'}

    Examples

    >>> from pyccel.types.ast import Declare, Variable
    >>> Declare('int', Variable('int', 'n'))
    Declare(NativeInteger(), (n,), None)
    >>> Declare('double', Variable('double', 'x'), intent='out')
    Declare(NativeDouble(), (x,), out)
    """

    def __new__(cls, dtype, variables, intent=None):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")
        # the following is not working for other concept than Variable
        # needed for example for MPI atoms
        if isinstance(variables, Variable):
            variables = [variables]
#        if not isinstance(variables, (list, tuple)):
#            variables = [variables]
        for var in variables:
            if not isinstance(var, Variable):
                raise TypeError("var must be of type Variable")
            if var.dtype != dtype:
                raise ValueError("All variables must have the same dtype")
        variables = Tuple(*variables)
        if intent:
            if not(intent in ['in', 'out', 'inout']):
                raise ValueError("intent must be one among {'in', 'out', 'inout'}")
        return Basic.__new__(cls, dtype, variables, intent)

    @property
    def dtype(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

    @property
    def intent(self):
        return self._args[2]

class Break(Basic):
    """Represents a break in the code."""
    pass

class Continue(Basic):
    """Represents a continue in the code."""
    pass

# TODO: improve with __new__ from Function and add example
class Len(Function):
    """
    Represents a 'len' expression in the code.
    """
    # TODO : remove later
    def __str__(self):
        return "len"

    def __new__(cls, rhs):
        return Basic.__new__(cls, rhs)

    @property
    def rhs(self):
        return self._args[0]

# TODO add example
class Shape(Basic):
    """Represents a 'shape' call in the code.

    lhs : list Expr
        list of assignable objects

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Shape
    """
    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        outputs = ', '.join(sstr(i) for i in self.lhs)
        return '{1} := shape({0})'.format(self.rhs, outputs)

# TODO: add example
class Min(Function):
    """Represents a 'min' expression in the code."""
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

# TODO: add example
class Max(Function):
    """Represents a 'max' expression in the code."""
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

# TODO: add example
class Mod(Function):
    """Represents a 'mod' expression in the code."""
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

# TODO: improve with __new__ from Function and add example
class Dot(Function):
    """
    Represents a 'dot' expression in the code.

    expr_l: variable
        first variable
    expr_r: variable
        second variable
    """
    def __new__(cls, expr_l, expr_r):
        return Basic.__new__(cls, expr_l, expr_r)

    @property
    def expr_l(self):
        return self.args[0]

    @property
    def expr_r(self):
        return self.args[1]

# TODO: treat as a Function
# TODO: add example
class Sign(Basic):

    def __new__(cls,expr):
        return Basic.__new__(cls, expr)

    @property
    def rhs(self):
        return self.args[0]

class Zeros(Basic):
    """Represents variable assignment using numpy.zeros for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int, list, tuple
        int or list of integers

    grid : Range, Tensor
        ensures a one-to-one representation of the array.

    Examples

    >>> from pyccel.types.ast import Variable, Zeros
    >>> n = Variable('int', 'n')
    >>> m = Variable('int', 'm')
    >>> x = Variable('int', 'x')
    >>> Zeros(x, (n,m))
    x := 0
    >>> y = Variable('bool', 'y')
    >>> Zeros(y, (n,m))
    y := False
    """
    # TODO improve in the spirit of assign
    def __new__(cls, lhs, shape=None, grid=None):
        lhs   = sympify(lhs)

        if shape:
            if isinstance(shape, list):
                # this is a correction. otherwise it is not working on LRZ
                if isinstance(shape[0], list):
                    shape = Tuple(*(sympify(i) for i in shape[0]))
                else:
                    shape = Tuple(*(sympify(i) for i in shape))
            elif isinstance(shape, int):
                shape = Tuple(sympify(shape))
            elif isinstance(shape,Len):
                shape = shape.str
            elif isinstance(shape, Basic):
                # TODO do we keep this?
                shape = str(shape)
            else:
                shape = shape

        if grid:
            if not isinstance(grid, (Range, Tensor)):
                raise TypeError('Expecting a Range or Tensor object.')

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        return Basic.__new__(cls, lhs, shape, grid)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.init_value))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def shape(self):
        if self._args[1]:
            return self._args[1]
        else:
            ranges = self.grid.ranges
            sh = [r.size for r in ranges]
            return Tuple(*(i for i in sh))

    @property
    def grid(self):
        return self._args[2]

    @property
    def init_value(self):
        dtype = self.lhs.dtype
        if isinstance(dtype, NativeInteger):
            value = 0
        elif isinstance(dtype, NativeFloat):
            value = 0.0
        elif isinstance(dtype, NativeDouble):
            value = 0.0
        elif isinstance(dtype, NativeComplex):
            value = 0.0
        elif isinstance(dtype, NativeBool):
            value = BooleanFalse()
        else:
            raise TypeError('Unknown type')
        return value

class Ones(Zeros):
    """
    Represents variable assignment using numpy.ones for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers

    Examples

    >>> from pyccel.types.ast import Variable, Ones
    >>> n = Variable('int', 'n')
    >>> m = Variable('int', 'm')
    >>> x = Variable('int', 'x')
    >>> Ones(x, (n,m))
    x := 1
    >>> y = Variable('bool', 'y')
    >>> Ones(y, (n,m))
    y := True
    """
    @property
    def init_value(self):
        dtype = self.lhs.dtype
        if isinstance(dtype, NativeInteger):
            value = 1
        elif isinstance(dtype, NativeFloat):
            value = 1.0
        elif isinstance(dtype, NativeDouble):
            value = 1.0
        elif isinstance(dtype, NativeComplex):
            value = 1.0
        elif isinstance(dtype, NativeBool):
            value = BooleanTrue()
        else:
            raise TypeError('Unknown type')
        return value

# TODO: add example
class Array(Basic):
    """Represents variable assignment using numpy.array for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        Sympy object representing the rhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers
    """
    def __new__(cls, lhs,rhs,shape):
        lhs   = sympify(lhs)


        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        if not isinstance(rhs, (list, ndarray)):
            raise TypeError("cannot assign rhs of type %s." % type(rhs))
        if not isinstance(shape, tuple):
            raise TypeError("shape must be of type tuple")


        return Basic.__new__(cls, lhs, rhs,shape)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := 0'.format(sstr(self.lhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def shape(self):
        return self._args[2]

# TODO add examples
class ZerosLike(Basic):
    """Represents variable assignment using numpy.zeros_like for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Variable
        the input variable

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Zeros, ZerosLike
    >>> n,m,x = symbols('n,m,x')
    >>> y = Zeros(x, (n,m))
    >>> z = ZerosLike(y)
    """
    # TODO improve in the spirit of assign
    def __new__(cls, lhs, rhs):
        if isinstance(lhs, str):
            lhs = Symbol(lhs)
        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, \
                      Indexed, Idx, Variable)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        return Basic.__new__(cls, lhs, rhs)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := 0'.format(sstr(self.lhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

# TODO: treat as a function
class Print(Basic):
    """Represents a print function in the code.

    expr : sympy expr
        The expression to return.

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """

    def __new__(cls, expr):
        if not isinstance(expr, list):
            expr = sympify(expr)
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

class Del(Basic):
    """Represents a memory deallocation in the code.

    variables : list, tuple
        a list of pyccel variables

    Examples

    >>> from pyccel.types.ast import Del, Variable
    >>> x = Variable('float', 'x', rank=2, shape=(10,2), allocatable=True)
    >>> Del([x])
    Del([x])
    """

    def __new__(cls, expr):
        # TODO: check that the variable is allocatable
        if not iterable(expr):
            expr = Tuple(expr)
        return Basic.__new__(cls, expr)

    @property
    def variables(self):
        return self._args[0]

# TODO: use dict instead of list for options
class Sync(Basic):
    """Represents a memory sync in the code.

    variables : list, tuple
        a list of pyccel variables

    master: Basic
        a master object running sync

    action: str
        the action to apply in parallel (ex: 'reduce')

    options: list
        a list of additional options (ex: '+' in case of reduce)

    Examples

    >>> from pyccel.types.ast import Sync, Variable
    >>> x = Variable('float', 'x', rank=2, shape=(10,2), allocatable=True)
    >>> Sync([x])
    Sync([x])
    >>> master = Variable('int', 'master')
    >>> Sync([x], master=master)
    Sync([x], master)
    """

    def __new__(cls, expr, master=None, action=None, options=[]):
        if not iterable(expr):
            expr = Tuple(expr)
        if action:
            if not isinstance(action, str):
                raise TypeError('Expecting a string')
        if not isinstance(options, list):
            raise TypeError('Expecting a list')

        return Basic.__new__(cls, expr, master, action, options)

    @property
    def variables(self):
        return self._args[0]

    @property
    def master(self):
        return self._args[1]

    @property
    def action(self):
        return self._args[2]

    @property
    def options(self):
        return self._args[3]


class EmptyLine(Basic):
    """Represents a EmptyLine in the code.

    text : str
       the comment line

    Examples

    >>> from pyccel.types.ast import EmptyLine
    >>> EmptyLine()

    """

    def __new__(cls):
        return Basic.__new__(cls)

    def _sympystr(self, printer):
        return '\n'

class Comment(Basic):
    """Represents a Comment in the code.

    text : str
       the comment line

    Examples

    >>> from pyccel.types.ast import Comment
    >>> Comment('this is a comment')
    Comment(this is a comment)
    """

    def __new__(cls, text):
        return Basic.__new__(cls, text)

    @property
    def text(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '# {0}'.format(sstr(self.text))

class AnnotatedComment(Basic):
    """Represents a Annotated Comment in the code.

    accel : str
       accelerator id. One among {'omp', 'acc'}

    txt: str
        statement to print

    Examples

    >>> from pyccel.types.ast import AnnotatedComment
    >>> AnnotatedComment('omp', 'parallel')
    AnnotatedComment(omp, parallel)
    """
    def __new__(cls, accel, txt):
        return Basic.__new__(cls, accel, txt)

    @property
    def accel(self):
        return self._args[0]

    @property
    def txt(self):
        return self._args[1]

class IndexedVariable(IndexedBase):
    """
    Represents an indexed variable, like x in x[i], in the code.

    Examples

    >>> from sympy import symbols, Idx
    >>> from pyccel.types.ast import IndexedVariable
    >>> A = IndexedVariable('A'); A
    A
    >>> type(A)
    <class 'pyccel.types.ast.IndexedVariable'>

    When an IndexedVariable object receives indices, it returns an array with named
    axes, represented by an IndexedElement object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'pyccel.types.ast.IndexedElement'>

    The IndexedVariable constructor takes an optional shape argument.  If given,
    it overrides any shape information in the indices. (But not the index
    ranges!)

    >>> m, n, o, p = symbols('m n o p', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> A[i, j].shape
    (m, n)
    >>> B = IndexedVariable('B', shape=(o, p))
    >>> B[i, j].shape
    (m, n)

    **todo:** fix bug. the last result must be : (o,p)
    """

    def __new__(cls, label, shape=None, dtype=None, **kw_args):
        obj = IndexedBase.__new__(cls, label, shape=shape, **kw_args)
        obj._dtype = dtype
        return obj

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
            return IndexedElement(self, *indices, **kw_args)
        else:
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return IndexedElement(self, indices, **kw_args)

    @property
    def dtype(self):
        return self._dtype


class IndexedElement(Indexed):
    """
    Represents a mathematical object with indices.

    Examples

    >>> from sympy import symbols, Idx
    >>> from pyccel.types.ast import IndexedVariable
    >>> i, j = symbols('i j', cls=Idx)
    >>> IndexedElement('A', i, j)
    A[i, j]

    It is recommended that ``IndexedElement`` objects be created via ``IndexedVariable``:

    >>> from pyccel.types.ast import IndexedElement
    >>> A = IndexedVariable('A')
    >>> IndexedElement('A', i, j) == A[i, j]
    False

    **todo:** fix bug. the last result must be : True
    """
    def __new__(cls, base, *args, **kw_args):
        from sympy.utilities.misc import filldedent
        from sympy.tensor.array.ndim_array import NDimArray
        from sympy.matrices.matrices import MatrixBase

        if not args:
            raise IndexException("Indexed needs at least one index.")
        if isinstance(base, (string_types, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base, IndexedBase):
            raise TypeError(filldedent("""
                Indexed expects string, Symbol, or IndexedBase as base."""))
        args = list(map(sympify, args))
        if isinstance(base, (NDimArray, collections.Iterable, Tuple, MatrixBase)) and all([i.is_number for i in args]):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]

        return Expr.__new__(cls, base, *args, **kw_args)

    @property
    def rank(self):
        """
        Returns the rank of the ``IndexedElement`` object.

        Examples

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j, k, l, m = symbols('i:m', cls=Idx)
        >>> Indexed('A', i, j).rank
        2
        >>> q = Indexed('A', i, j, k, l, m)
        >>> q.rank
        5
        >>> q.rank == len(q.indices)
        True

        """
        n = 0
        for a in self.args[1:]:
            if not(isinstance(a, Slice)):
                n += 1
        return n

    @property
    def dtype(self):
        return self.base.dtype

# TODO check that args are integers
class Slice(Basic):
    """Represents a slice in the code.

    start : Symbol or int
        starting index

    end : Symbol or int
        ending index

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Slice
    >>> m, n = symbols('m, n', integer=True)
    >>> Slice(m,n)
    m : n
    >>> Slice(None,n)
     : n
    >>> Slice(m,None)
    m :
    """
    # TODO add step

    def __new__(cls, start, end):
        return Basic.__new__(cls, start, end)

    @property
    def start(self):
        return self._args[0]

    @property
    def end(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.start is None:
            start = ''
        else:
            start = sstr(self.start)
        if self.end is None:
            end = ''
        else:
            end = sstr(self.end)
        return '{0} : {1}'.format(start, end)

class If(Basic):
    """Represents a if statement in the code.

    args :
        every argument is a tuple and
        is defined as (cond, expr) where expr is a valid ast element
        and cond is a boolean test.

    Examples

    >>> from sympy import Symbol
    >>> from pyccel.types.ast import Assign, If
    >>> n = Symbol('n')
    >>> If(((n>1), [Assign(n,n-1)]), (True, [Assign(n,n+1)]))
    If(((n>1), [Assign(n,n-1)]), (True, [Assign(n,n+1)]))
    """
    # TODO add step
    def __new__(cls, *args):
        # (Try to) sympify args first
        newargs = []
        for ce in args:
            cond = ce[0]
            if not isinstance(cond, (bool, Relational, Boolean)):
                raise TypeError(
                    "Cond %s is of type %s, but must be a Relational,"
                    " Boolean, or a built-in bool." % (cond, type(cond)))
            newargs.append(ce)

        return Basic.__new__(cls, *newargs)

class MultiAssign(Basic):
    """Represents a multiple assignment statement in the code.
    In Fortran, this will be interpreted as a subroutine call.

    lhs : list Expr
        list of assignable objects
    rhs : Function
        function call expression

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import MultiAssign
    >>> from pyccel.types.ast import Assign, Variable, FunctionDef
    >>> x, y, z, t = symbols('x, y, z, t')
    >>> args        = [Variable('float', x), Variable('float', y)]
    >>> results     = [Variable('float', z), Variable('float', t)]
    >>> body        = [Assign(z,x+y), Assign(t,x*y)]
    >>> local_vars  = []
    >>> global_vars = []
    >>> f = FunctionDef('f', args, results, body, local_vars, global_vars)
    >>> MultiAssign((z,t), f)
    z, t := FunctionDef(f, (x, y), (z, t), [z := x + y, t := x*y], [], [])
    """
    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr    = printer.doprint
        rhs     = sstr(self.rhs)
        outputs = ', '.join(sstr(i) for i in self.lhs)
        return '{0} := {1}'.format(outputs, rhs)

# TODO: to rewrite
class Thread(Basic):
    """Represents a thread function for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    Examples

    """

    def __new__(cls, lhs):
        lhs   = sympify(lhs)

        # Tuple of things that can be on the lhs of an assignment
        if not isinstance(lhs, Symbol):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        return Basic.__new__(cls, lhs)

    @property
    def lhs(self):
        return self._args[0]

# TODO: to rewrite
class ThreadID(Thread):
    """Represents a get thread id for code generation.
    """
    pass

# TODO: to rewrite
class ThreadsNumber(Thread):
    """Represents a get threads number for code generation.
    """
    pass

# TODO: remove Len from here
class Stencil(Basic):
    """Represents variable assignment using a stencil for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers

    step : int or list of integers

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Stencil
    >>> x, y, z = symbols('x, y, z')
    >>> m, n, p, q = symbols('m n p q', integer=True)
    >>> Stencil(x, n, p)
    Stencil(x, n, p)
    >>> Stencil(y, (n,m), (p,q))
    Stencil(y, (n, m), (p, q))
    """

    # TODO improve in the spirit of assign
    def __new__(cls, lhs, shape, step):
        # ...
        def format_entry(s_in):
            if isinstance(s_in, list):
                # this is a correction. otherwise it is not working on LRZ
                if isinstance(s_in[0], list):
                    s_out = Tuple(*(sympify(i) for i in s_in[0]))
                else:
                    s_out = Tuple(*(sympify(i) for i in s_in))
            elif isinstance(s_in, int):
                s_out = Tuple(sympify(s_in))
            elif isinstance(s_in, Basic) and not isinstance(s_in,Len):
                s_out = str(s_in)
            elif isinstance(s_in,Len):
                s_our = s_in.str
            else:
                s_out = s_in
            return s_out
        # ...

        # ...
        lhs   = sympify(lhs)
        shape = format_entry(shape)
        step  = format_entry(step)
        # ...

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        return Basic.__new__(cls, lhs, shape, step)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def shape(self):
        return self._args[1]

    @property
    def step(self):
        return self._args[2]

class FunctionHeader(Basic):
    """Represents function/subroutine header in the code.

    func: str
        function/subroutine name

    dtypes: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr)

    results: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr)

    kind: str
        'function' or 'procedure'. default value: 'function'

    Examples

    >>> from pyccel.types.ast import FunctionHeader
    >>> FunctionHeader('f', ['double'])
    FunctionHeader(f, [(NativeDouble(), [])])
    >>> FunctionHeader('mpi_dims_create', ['int', 'int', ('int', [Slice(None,None)])], results=['int'])
    FunctionHeader(mpi_dims_create, [(NativeInteger(), []), (NativeInteger(), []), (int, [ : ])], [(NativeInteger(), [])], function)
    """

    def __new__(cls, func, dtypes, results=None, kind='function'):
        if not(iterable(dtypes)):
            raise TypeError("Expecting dtypes to be iterable.")

        types = []
        for d in dtypes:
            if isinstance(d, str):
                types.append((datatype(d), []))
            elif isinstance(d, DataType):
                types.append((d, []))
            elif isinstance(d, (tuple, list)):
                if not(len(d) == 2):
                    raise ValueError("Expecting exactly two entries.")
                types.append(d)
            else:
                raise TypeError("Wrong element in dtypes.")

        r_types = []
        if results:
            if not(iterable(results)):
                raise TypeError("Expecting results to be iterable.")

            r_types = []
            for d in results:
                if isinstance(d, str):
                    r_types.append((datatype(d), []))
                elif isinstance(d, DataType):
                    r_types.append((d, []))
                elif isinstance(d, (tuple, list)):
                    if not(len(d) == 2):
                        raise ValueError("Expecting exactly two entries.")
                    r_types.append(d)
                else:
                    raise TypeError("Wrong element in r_types.")

        if not isinstance(kind, str):
            raise TypeError("Expecting a string for kind.")

        if not (kind in ['function', 'procedure']):
            print( kind)
            raise ValueError("kind must be one among {'function', 'procedure'}")

        return Basic.__new__(cls, func, types, r_types, kind)

    @property
    def func(self):
        return self._args[0]

    @property
    def dtypes(self):
        return self._args[1]

    @property
    def results(self):
        return self._args[2]

    @property
    def kind(self):
        return self._args[3]

class MethodHeader(FunctionHeader):
    """Represents method header in the code.

    name: iterable
        method name as a list/tuple

    dtypes: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr)

    results: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr)

    Examples

    >>> from pyccel.types.ast import MethodHeader
    >>> m = MethodHeader(('point', 'rotate'), ['double'])
    >>> m
    MethodHeader((point, rotate), [(NativeDouble(), [])], [])
    >>> m.name
    'point.rotate'
    """

    def __new__(cls, name, dtypes, results=None):
        if not isinstance(name, (list, tuple)):
            raise TypeError("Expecting a list/tuple of strings.")

        if not(iterable(dtypes)):
            raise TypeError("Expecting dtypes to be iterable.")

        types = []
        for d in dtypes:
            if isinstance(d, str):
                types.append((datatype(d), []))
            elif isinstance(d, DataType):
                types.append((d, []))
            elif isinstance(d, (tuple, list)):
                if not(len(d) == 2):
                    raise ValueError("Expecting exactly two entries.")
                types.append(d)
            else:
                raise TypeError("Wrong element in dtypes.")

        r_types = []
        if results:
            if not(iterable(results)):
                raise TypeError("Expecting results to be iterable.")

            r_types = []
            for d in results:
                if isinstance(d, str):
                    r_types.append((datatype(d), []))
                elif isinstance(d, DataType):
                    r_types.append((d, []))
                elif isinstance(d, (tuple, list)):
                    if not(len(d) == 2):
                        raise ValueError("Expecting exactly two entries.")
                    r_types.append(d)
                else:
                    raise TypeError("Wrong element in r_types.")

        return Basic.__new__(cls, name, types, r_types)

    @property
    def name(self):
        _name = self._args[0]
        if isinstance(_name, str):
            return _name
        else:
            return '.'.join(str(n) for n in _name)

    @property
    def dtypes(self):
        return self._args[1]

    @property
    def results(self):
        return self._args[2]

class ClassHeader(Basic):
    """Represents class header in the code.

    name: str
        class name

    options: str, list, tuple
        a list of options

    Examples

    >>> from pyccel.types.ast import ClassHeader
    >>> ClassHeader('Matrix', ('abstract', 'public'))
    ClassHeader(Matrix, (abstract, public))
    """

    def __new__(cls, name, options):
        if not(iterable(options)):
            raise TypeError("Expecting options to be iterable.")

        return Basic.__new__(cls, name, options)

    @property
    def name(self):
        return self._args[0]

    @property
    def options(self):
        return self._args[1]
