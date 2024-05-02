#!/usr/bin/python
# -*- coding: utf-8 -*-

import importlib

from collections.abc import Iterable

from sympy import cache
from sympy import sympify
from sympy import Add, Mul, Pow as sp_Pow
from sympy import Integral, Symbol, Tuple
from sympy import Lambda, preorder_traversal
from sympy import Integer as sp_Integer
from sympy import Float as sp_Float, Rational as sp_Rational
from sympy import preorder_traversal

from sympy.simplify.radsimp   import fraction
from sympy.core.compatibility import with_metaclass
from sympy.core.compatibility import is_sequence
from sympy.core.compatibility import string_types
from sympy.core.assumptions   import StdFactKB
from sympy.core.operations    import LatticeOp
from sympy.core.relational    import Equality, Relational
from sympy.core.relational    import Eq, Ne, Lt, Gt, Le, Ge
from sympy.core.singleton     import Singleton, S
from sympy.core.function      import Function, Application
from sympy.core.function      import Derivative, UndefinedFunction
from sympy.core.function      import _coeff_isneg
from sympy.core.numbers       import ImaginaryUnit
from sympy.core.basic         import Atom
from sympy.core.expr          import Expr, AtomicExpr
from sympy.logic.boolalg      import And, Boolean, Not, Or, true, false
from sympy.logic.boolalg      import Boolean, BooleanTrue, BooleanFalse
from sympy.logic.boolalg      import BooleanFunction
from sympy.tensor             import Idx, Indexed, IndexedBase

from sympy.matrices.matrices            import MatrixBase
from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
from sympy.tensor.array.ndim_array      import NDimArray
from sympy.utilities.iterables          import iterable
from sympy.utilities.misc               import filldedent


from .basic import Basic
from .builtins import Enumerate, Len, List, Map, Range, Zip
from .datatypes import (datatype, DataType, CustomDataType, NativeSymbol,
                        NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeRange, NativeTensor, NativeString,
                        NativeGeneric, default_precision)

from .functionalexpr import GeneratorComprehension as GC
from .functionalexpr import FunctionalFor

# TODO [YG, 12.03.2020]: Move non-Python constructs to other modules
# TODO [YG, 12.03.2020]: Rename classes to avoid name clashes in pyccel/ast
# NOTE: commented-out symbols are never used in Pyccel
__all__ = (
    'AddOp',
    'AliasAssign',
    'AnnotatedComment',
    'Argument',
    'AsName',
    'Assert',
    'Assign',
    'AstError',
    'AstFunctionResultError',
    'AugAssign',
    'Block',
    'Break',
    'ClassDef',
    'CodeBlock',
    'Comment',
    'CommentBlock',
    'Concatenate',
    'ConstructorCall',
    'Continue',
    'Declare',
    'Del',
    'DivOp',
    'Dlist',
    'DoConcurrent',
    'DottedName',
    'DottedVariable',
    'EmptyLine',
    'ErrorExit',
    'Eval',
    'Exit',
    'F2PYFunctionDef',
    'For',
    'ForAll',
    'ForIterator',
    'FunctionCall',
    'FunctionDef',
    'GetDefaultFunctionArg',
    'If',
    'IfTernaryOperator',
    'Import',
    'IndexedElement',
    'IndexedVariable',
    'Interface',
    'Is',
    'IsNot',
    'Load',
    'ModOp',
    'Module',
    'MulOp',
    'NativeOp',
    'NewLine',
    'Nil',
    'ParallelBlock',
    'ParallelRange',
    'Pass',
    'Pow',
    'Product',
    'Program',
    'PythonFunction',
    'Random',
    'Return',
    'SeparatorComment',
    'Slice',
    'String',
    'SubOp',
    'Subroutine',
    'SumFunction',
    'SymbolicAssign',
    'SymbolicPrint',
    'SympyFunction',
    'Tensor',
    'Tile',
    'TupleImport',
    'ValuedArgument',
    'ValuedVariable',
    'Variable',
    'Void',
    'VoidFunction',
    'While',
    'With',
    '_atomic',
#    'allocatable_like',
    'collect_vars',
    'create_variable',
    'extract_subexpressions',
#    'float2int',
    'get_assigned_symbols',
    'get_initial_value',
    'get_iterable_ranges',
    'inline',
    'int2float',
#    'is_simple_assign',
    'local_sympify',
#    'operator',
#    'op_registry',
    'subs'
)

#==============================================================================
local_sympify = {
    'N'    : Symbol('N'),
    'S'    : Symbol('S'),
    'zeros': Symbol('zeros'),
    'ones' : Symbol('ones'),
    'Point': Symbol('Point')
}

#==============================================================================
class AstError(Exception):
    pass

class AstFunctionResultError(AstError):
    def __init__(self, var):
        if isinstance(var, (list, tuple, Tuple)):
            var = ', '.join(str(i) for i in var)

        msg = 'Found allocatable result(s) that is/are not inout [{}]'.format(var)

        # Call the base class constructor with the parameters it needs
        super(AstFunctionResultError, self).__init__(msg)



# TODO - add EmptyStmt => empty lines
#      - update code examples
#      - add examples
#      - Function case
#      - AnnotatedComment case
#      - use Tuple after checking the object is iterable:'funcs=Tuple(*funcs)'
#      - add a new Idx that uses Variable instead of Symbol


def subs(expr, new_elements):
    """
    Substitutes old for new in an expression after sympifying args.

    new_elements : list of tuples like [(x,2)(y,3)]
    """

    if len(list(new_elements)) == 0:
        return expr
    if isinstance(expr, (list, tuple, Tuple)):
        return [subs(i, new_elements) for i in expr]

    elif isinstance(expr, While):
        test = subs(expr.test, new_elements)
        body = subs(expr.body, new_elements)
        return While(test, body)

    elif isinstance(expr, For):
        target = subs(expr.target, new_elements)
        it = subs(expr.iterable, new_elements)
        target = expr.target
        it = expr.iterable
        body = subs(expr.body, new_elements)
        return For(target, it, body)

    elif isinstance(expr, If):
        args = []
        for block in expr.args:
            test = block[0]
            stmts = block[1]
            t = subs(test, new_elements)
            s = subs(stmts, new_elements)
            args.append((t, s))
        return If(*args)

    elif isinstance(expr, Return):

        for i in new_elements:
            expr = expr.subs(i[0],i[1])
        return expr

    elif isinstance(expr, Assign):
        new_expr = expr.subs(new_elements)
        new_expr.set_fst(expr.fst)
        return new_expr
    elif isinstance(expr, Expr):
        return expr.subs(new_elements)

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

    if isinstance(expr, (Variable, IndexedVariable, IndexedElement)):
        return expr
    elif isinstance(expr, str):
        # if the rhs is a string
        return expr
    elif isinstance(expr, Expr):
        args = [expr]
        while args:
            a = args.pop()
            # XXX: This is a hack to support non-Basic args
            if isinstance(a, string_types):
                continue

            if a.is_Mul:
                if _coeff_isneg(a):
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                (n, d) = fraction(a)
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
                for (i, ai) in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                    else:
                        args.append(ai)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if a.is_Mul or a.is_Pow or a.is_Function or isinstance(a,
                    Derivative) or isinstance(a, Integral):

                o = Symbol(a.func.__name__.upper())
            if not a.is_Symbol and not isinstance(a, (IndexedElement,
                    Function)):
                args.extend(a.args)
            if isinstance(a, Function):
                if verbose:
                    print('Functions not yet available')
                return None
            elif isinstance(a, (Variable, IndexedVariable,
                            IndexedElement)):
                return a
            elif a.is_Symbol:
                raise TypeError('Found an unknown symbol {0}'.format(str(a)))
    else:
        raise TypeError('Unexpected type {0}'.format(type(expr)))



def _atomic(e, cls=None,ignore=()):

    """Return atom-like quantities as far as substitution is
    concerned: Functions and DottedVarviables, Variables. we don't
    return atoms that are inside such quantities too
    """

    pot = preorder_traversal(e)
    seen = []
    atoms_ = []
    if cls is None:
        cls = (Application, DottedVariable, Variable,
               IndexedVariable,IndexedElement)

    for p in pot:
        if p in seen or isinstance(p, ignore):
            pot.skip()
            continue
        seen.append(p)
        if isinstance(p, cls):
            pot.skip()
            atoms_.append(p)

    return atoms_



def extract_subexpressions(expr):
    """this function takes an expression and returns a list
      of statements if this expression contains sub expressions that need
      to be evaluated outside of the expression


      expr : Add, Mul, Pow, Application

    """

    stmts = []
    cls   = (Add, Mul, sp_Pow, And,
             Or, Eq, Ne, Lt, Gt,
             Le, Ge)

    id_cls = (Symbol, Indexed, IndexedBase,
              DottedVariable, sp_Float, sp_Integer,
              sp_Rational, ImaginaryUnit,Boolean,
              BooleanTrue, BooleanFalse, String,
              ValuedArgument, Nil, List)

    func_names = ('diag', 'empty', 'zip', 'enumerate')
    #TODO put only imported functions
    def substitute(expr):
        if isinstance(expr, id_cls):
            return expr
        if isinstance(expr, cls):
            args = expr.args
            args = [substitute(arg) for arg in args]
            return expr.func(*args, evaluate=False)
        elif isinstance(expr, Application):
            args = substitute(expr.args)

            if str(expr.func) in func_names:
                var = create_variable(expr)
                expr = expr.func(*args, evaluate=False)
                expr = Assign(var, expr)
                stmts.append(expr)

                return var
            else:
                expr = expr.func(*args, evaluate=False)
                return expr
        elif isinstance(expr, GC):
            stmts.append(expr)
            return expr.lhs
        elif isinstance(expr,IfTernaryOperator):
            var = create_variable(expr)
            new = Assign(var, expr)
            new.set_fst(expr.fst)
            stmts.append(new)
            return var
        elif isinstance(expr, List):
            args = []
            for i in expr:
                args.append(substitute(i))

            return List(*args, sympify=False)

        elif isinstance(expr, (Tuple, tuple, list)):
            args = []

            for i in expr:
                args.append(substitute(i))
            return args

        else:
            raise TypeError('statment {} not supported yet'.format(type(expr)))


    new_expr  = substitute(expr)
    return stmts, new_expr



def collect_vars(ast):
    """ collect variables in order to be declared"""
    #TODO use the namespace to get the declared variables
    variables = {}
    def collect(stmt):

        if isinstance(stmt, Variable):
            if not isinstance(stmt.name, DottedName):
                variables[stmt.name] = stmt
        elif isinstance(stmt, (tuple, Tuple, list)):
            for i in stmt:
                collect(i)
        if isinstance(stmt, For):
            collect(stmt.target)
            collect(stmt.body)
        elif isinstance(stmt, FunctionalFor):
            collect(stmt.lhs)
            collect(stmt.loops)
        elif isinstance(stmt, If):
            collect(stmt.bodies)
        elif isinstance(stmt, (While, CodeBlock)):
            collect(stmt.body)
        elif isinstance(stmt, (Assign, AliasAssign, AugAssign)):
            collect(stmt.lhs)
            if isinstance(stmt.rhs, (Linspace, Diag, Where)):
                collect(stmt.rhs.index)



    collect(ast)
    return variables.values()

def inline(func, args):
        local_vars = func.local_vars
        body = func.body
        body = subs(body, zip(func.arguments, args))
        return Block(str(func.name), local_vars, body)


def int2float(expr):
    return expr

def float2int(expr):
    return expr

def create_variable(expr):
    """."""

    import numpy as np
    try:
        name = 'Dummy_' + str(abs(hash(expr)
                                  + np.random.randint(500)))[-4:]
    except:
        name = 'Dymmy_' + str(abs(np.random.randint(500)))[-4:]

    return Symbol(name)

class Pow(sp_Pow):

    def _eval_subs(self, old, new):
        args = self.args
        args_ = [self.base._subs(old, new),self.exp._subs(old, new)]
        args  = [args_[i] if args_[i] else args[i] for i in range(len(args))]
        expr = Pow(args[0], args[1], evaluate=False)
        return expr

    def _eval_evalf(self,prec):
        return sp_Pow(self.base,self.exp).evalf(prec)



class DottedName(Basic):

    """
    Represents a dotted variable.

    Examples

    >>> from pyccel.ast.core import DottedName
    >>> DottedName('matrix', 'n_rows')
    matrix.n_rows
    >>> DottedName('pyccel', 'stdlib', 'parallel')
    pyccel.stdlib.parallel
    """

    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

    @property
    def name(self):
        return self._args

    def __str__(self):
        return """.""".join(str(n) for n in self.name)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return """.""".join(sstr(n) for n in self.name)


class AsName(Basic):

    """
    Represents a renaming of a variable, used with Import.

    Examples

    >>> from pyccel.ast.core import AsName
    >>> AsName('new', 'old')
    new as old
    """

    def __new__(cls, name, target):

        # TODO check

        return Basic.__new__(cls, name, target)

    @property
    def name(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} as {1}'.format(sstr(self.name), sstr(self.target))


class Dlist(Basic):

    """ this is equivalent to the zeros function of numpy arrays for the python list.

    value : Expr
           a sympy expression which represents the initilized value of the list

    shape : the shape of the array
    """

    def __new__(cls, val, length):
        return Basic.__new__(cls, val, length)

    @property
    def val(self):
        return self._args[0]

    @property
    def length(self):
        return self._args[1]


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
    >>> from pyccel.ast.core import Assign
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

    def __new__(
        cls,
        lhs,
        rhs,
        strict=False,
        status=None,
        like=None,
        ):
        cls._strict = strict

        if strict:
            lhs = sympify(lhs, locals=local_sympify)
            rhs = sympify(rhs, locals=local_sympify)

            # Tuple of things that can be on the lhs of an assignment

            assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed,
                          Idx)

            # if not isinstance(lhs, assignable):
            #    raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
            # Indexed types implement shape, but don't define it until later. This
            # causes issues in assignment validation. For now, matrices are defined
            # as anything with a shape that is not an Indexed

            lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs,
                    Indexed)
            rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs,
                    Indexed)

            # If lhs and rhs have same structure, then this assignment is ok

            if lhs_is_mat:
                if not rhs_is_mat:
                    raise ValueError('Cannot assign a scalar to a matrix.')
                elif lhs.shape != rhs.shape:
                    raise ValueError("Dimensions of lhs and rhs don't align.")
            elif rhs_is_mat and not lhs_is_mat:
                raise ValueError('Cannot assign a matrix to a scalar.')
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

    # TODO : remove

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

    @property
    def is_alias(self):
        """Returns True if the assignment is an alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        cond = isinstance(rhs, Variable) and rhs.rank > 0
        cond = cond or isinstance(rhs, IndexedElement)
        cond = cond or isinstance(rhs, IndexedVariable)
        cond = cond and isinstance(lhs, Symbol)
        cond = cond or isinstance(rhs, Variable) and rhs.is_pointer
        return cond

    @property
    def is_symbolic_alias(self):
        """Returns True if the assignment is a symbolic alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        if isinstance(lhs, Variable):
            return isinstance(lhs.dtype, NativeSymbol)
        elif isinstance(lhs, Symbol):
            if isinstance(rhs, Range):
                return True
            elif isinstance(rhs, Variable):
                return isinstance(rhs.dtype, NativeSymbol)
            elif isinstance(rhs, Symbol):
                return True

        return False


class CodeBlock(Basic):

    """Represents a list of stmt for code generation.
       we use it when a single statement in python
       produce multiple statement in the targeted language
    """

    def __new__(cls, body):
        ls = []
        for i in body:
            if isinstance(i, CodeBlock):
                ls += i.body
            else:
                ls.append(i)

        obj = Basic.__new__(cls, ls)
        if len(ls)>0 and isinstance(ls[-1], (Assign, AugAssign)):
            obj.set_fst(ls[-1].fst)
        return obj

    @property
    def body(self):
        return self._args[0]

    @property
    def lhs(self):
        return self.body[-1].lhs


class AliasAssign(Basic):

    """Represents aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    lhs : Symbol
        at this point we don't know yet all information about lhs, this is why a
        Symbol is the appropriate type.

    rhs : Variable, IndexedVariable, IndexedElement
        an assignable variable can be of any rank and any datatype, however its
        shape must be known (not None)

    Examples

    >>> from sympy import Symbol
    >>> from pyccel.ast.core import AliasAssign
    >>> from pyccel.ast.core import Variable
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x', rank=1, shape=[n])
    >>> y = Symbol('y')
    >>> AliasAssign(y, x)

    """

    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


class SymbolicAssign(Basic):

    """Represents symbolic aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    lhs : Symbol

    rhs : Range

    Examples

    >>> from sympy import Symbol
    >>> from pyccel.ast.core import SymbolicAssign
    >>> from pyccel.ast.core import Range
    >>> r = Range(0, 3)
    >>> y = Symbol('y')
    >>> SymbolicAssign(y, r)

    """

    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := {1}'.format(sstr(self.lhs), sstr(self.rhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


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


op_registry = {
    '+': AddOp(),
    '-': SubOp(),
    '*': MulOp(),
    '/': DivOp(),
    '%': ModOp(),
    }


def operator(op):
    """Returns the operator singleton for the given operator"""

    if op.lower() not in op_registry:
        raise ValueError('Unrecognized operator ' + op)
    return op_registry[op]


class AugAssign(Assign):

    r"""
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

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import AugAssign
    >>> s = Variable('int', 's')
    >>> t = Variable('int', 't')
    >>> AugAssign(s, '+', 2 * t + 1)
    s += 1 + 2*t
    """

    def __new__(
        cls,
        lhs,
        op,
        rhs,
        strict=False,
        status=None,
        like=None,
        ):
        cls._strict = strict
        if strict:
            lhs = sympify(lhs, locals=local_sympify)
            rhs = sympify(rhs, locals=local_sympify)

            # Tuple of things that can be on the lhs of an assignment

            assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed)
            if not isinstance(lhs, assignable):
                raise TypeError('Cannot assign to lhs of type %s.'
                                % type(lhs))

            # Indexed types implement shape, but don't define it until later. This
            # causes issues in assignment validation. For now, matrices are defined
            # as anything with a shape that is not an Indexed

            lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs,
                    Indexed)
            rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs,
                    Indexed)

            # If lhs and rhs have same structure, then this assignment is ok

            if lhs_is_mat:
                if not rhs_is_mat:
                    raise ValueError('Cannot assign a scalar to a matrix.'
                            )
                elif lhs.shape != rhs.shape:
                    raise ValueError("Dimensions of lhs and rhs don't align."
                            )
            elif rhs_is_mat and not lhs_is_mat:
                raise ValueError('Cannot assign a matrix to a scalar.')

        if isinstance(op, str):
            op = operator(op)
        elif op not in list(op_registry.values()):
            raise TypeError('Unrecognized Operator')

        return Basic.__new__(
            cls,
            lhs,
            op,
            rhs,
            status,
            like,
            )

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
    >>> from pyccel.ast.core import Assign, While
    >>> n = Symbol('n')
    >>> While((n>1), [Assign(n,n-1)])
    While(n > 1, (n := n - 1,))
    """

    def __new__(cls, test, body, local_vars=[]):
        test = sympify(test, locals=local_sympify)

        if iterable(body):
            body = CodeBlock((sympify(i, locals=local_sympify) for i in body))
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')
        return Basic.__new__(cls, test, body, local_vars)

    @property
    def test(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

    @property
    def local_vars(self):
        return self._args[2]


class With(Basic):

    """Represents a 'with' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    test : expression
        test condition given as a sympy expression
    body : sympy expr
        list of statements representing the body of the With statement.

    Examples

    """

    # TODO check prelude and epilog

    def __new__(
        cls,
        test,
        body,
        settings,
        ):
        test = sympify(test, locals=local_sympify)

        if iterable(body):
            body = CodeBlock((sympify(i, locals=local_sympify) for i in body))
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable')

        return Basic.__new__(cls, test, body, settings)

    @property
    def test(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

    @property
    def settings(self):
        return self._args[2]

    @property
    def block(self):
        methods = self.test.cls_base.methods
        for i in methods:
            if str(i.name) == '__enter__':
                enter = i
            elif str(i.name) == '__exit__':
                exit = i
        enter = inline(enter,[])
        exit =  inline(exit, [])

        # TODO check if enter is empty or not first

        body = enter.body.body
        body += self.body.body
        body += exit.body.body
        return Block('with', [], body)


class Product(Basic):

    """
    Represents a Product stmt.

    """

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            return args[0]
        return Basic.__new__(cls, *args)

    @property
    def elements(self):
        return self._args


class Tile(Range):

    """
    Representes a tile.

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Tile
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


class ParallelRange(Range):

    """
    Representes a parallel range using OpenMP/OpenACC.

    Examples

    >>> from pyccel.ast.core import Variable
    """

    pass


# TODO: implement it as an extension of sympy Tensor?

class Tensor(Basic):

    """
    Base class for tensor.

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Range, Tensor
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
            cond = isinstance(r, Variable) and isinstance(r.dtype,
                    (NativeRange, NativeTensor))
            cond = cond or isinstance(r, (Range, Tensor))

            if not cond:
                raise TypeError('non valid argument, given {0}'.format(type(r)))

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
        txt = ', '.join(sstr(n) for n in self.ranges)
        txt = 'Tensor({0}, name={1})'.format(txt, sstr(self.name))
        return txt


# TODO add a name to a block?

class Block(Basic):

    """Represents a block in the code. A block consists of the following inputs

    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    Examples

    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> Block([n, x], [Assign(x,2.*n + 1.), Assign(n, n + 1)])
    Block([n, x], [x := 1.0 + 2.0*n, n := 1 + n])
    """

    def __new__(
        cls,
        name,
        variables,
        body):
        if not isinstance(name, str):
            raise TypeError('name must be of type str')
        if not iterable(variables):
            raise TypeError('variables must be an iterable')
        for var in variables:
            if not isinstance(var, Variable):
                raise TypeError('Only a Variable instance is allowed.')
        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body, CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')
        return Basic.__new__(cls, name, variables, body)

    @property
    def name(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]


class ParallelBlock(Block):

    """
    Represents a parallel block in the code.
    In addition to block inputs, there is

    clauses: list
        a list of clauses

    Examples

    >>> from pyccel.ast.core import ParallelBlock
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> body = [Assign(x,2.*n + 1.), Assign(n, n + 1)]
    >>> variables = [x,n]
    >>> clauses = []
    >>> ParallelBlock(clauses, variables, body)
    # parallel
    x := 1.0 + 2.0*n
    n := 1 + n
    """

    _prefix = '#'

    def __new__(
        cls,
        clauses,
        variables,
        body,
        ):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        cls._clauses = clauses

        return Block.__new__(cls, variables, body)

    @property
    def clauses(self):
        return self._clauses

    @property
    def prefix(self):
        return self._prefix

    def _sympystr(self, printer):
        sstr = printer.doprint

        prefix = sstr(self.prefix)
        clauses = ' '.join('{0}'.format(sstr(i)) for i in self.clauses)
        body = '\n'.join('{0}'.format(sstr(i)) for i in self.body)

        code = '{0} parallel {1}\n{2}'.format(prefix, clauses, body)
        return code


class Module(Basic):

    """Represents a module in the code. A block consists of the following inputs

    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    funcs: list
        a list of FunctionDef instances

    classes: list
        a list of ClassDef instances

    imports: list, tuple
        list of needed imports

    Examples

    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
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

    def __new__(
        cls,
        name,
        variables,
        funcs,
        interfaces=[],
        classes=[],
        imports=[],
        ):
        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')
        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(funcs):
            raise TypeError('funcs must be an iterable')

        for i in funcs:
            if not isinstance(i, FunctionDef):
                raise TypeError('Only a FunctionDef instance is allowed.'
                                )

        if not iterable(classes):
            raise TypeError('classes must be an iterable')
        for i in classes:
            if not isinstance(i, ClassDef):
                raise TypeError('Only a ClassDef instance is allowed.')

        if not iterable(interfaces):
            raise TypeError('interfaces must be an iterable')
        for i in interfaces:
            if not isinstance(i, Interface):
                raise TypeError('Only a Inteface instance is allowed.')

        if not iterable(imports):
            raise TypeError('imports must be an iterable')
        imports = list(imports)
        for i in classes:
            imports += i.imports
        imports = set(imports)  # for unicity
        imports = Tuple(*imports, sympify=False)

        return Basic.__new__(
            cls,
            name,
            variables,
            funcs,
            interfaces,
            classes,
            imports,
            )

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
    def interfaces(self):
        return self._args[3]

    @property
    def classes(self):
        return self._args[4]

    @property
    def imports(self):
        return self._args[5]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]

    @property
    def body(self):
        return self.interfaces + self.funcs + self.classes


class Program(Basic):

    """Represents a Program in the code. A block consists of the following inputs

    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    funcs: list
        a list of FunctionDef instances

    classes: list
        a list of ClassDef instances

    body: list
        a list of statements

    imports: list, tuple
        list of needed imports

    modules: list, tuple
        list of needed modules

    Examples

    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
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

    def __new__(
        cls,
        name,
        variables,
        funcs,
        interfaces,
        classes,
        body,
        imports=[],
        modules=[],
        ):

        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')

        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(funcs):
            raise TypeError('funcs must be an iterable')
        for i in funcs:
            if not isinstance(i, FunctionDef):
                raise TypeError('Only a FunctionDef instance is allowed.'
                                )

        if not iterable(interfaces):
            raise TypeError('interfaces must be an iterable')
        for i in interfaces:
            if not isinstance(i, Interface):
                raise TypeError('Only a Interface instance is allowed.')

        if not iterable(body):
            raise TypeError('body must be an iterable')
        body = CodeBlock(body)

        if not iterable(classes):
            raise TypeError('classes must be an iterable')
        for i in classes:
            if not isinstance(i, ClassDef):
                raise TypeError('Only a ClassDef instance is allowed.')

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        for i in funcs:
            imports += i.imports
        for i in classes:
            imports += i.imports
        imports = set(imports)  # for unicity
        imports = Tuple(*imports, sympify=False)

        if not iterable(modules):
            raise TypeError('modules must be an iterable')

        # TODO
#        elif isinstance(stmt, list):
#            for s in stmt:
#                body += printer(s) + "\n"

        return Basic.__new__(
            cls,
            name,
            variables,
            funcs,
            interfaces,
            classes,
            body,
            imports,
            modules,
            )

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
    def interfaces(self):
        return self._args[3]

    @property
    def classes(self):
        return self._args[4]

    @property
    def body(self):
        return self._args[5]

    @property
    def imports(self):
        return self._args[6]

    @property
    def modules(self):
        return self._args[7]

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]


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
    >>> from pyccel.ast.core import Assign, For
    >>> i,b,e,s,x = symbols('i,b,e,s,x')
    >>> A = MatrixSymbol('A', 1, 3)
    >>> For(i, (b,e,s), [Assign(x,x-1), Assign(A[0, 1], x)])
    For(i, Range(b, e, s), (x := x - 1, A[0, 1] := x))
    """

    def __new__(
        cls,
        target,
        iter,
        body,
        local_vars = [],
        strict=True,
        ):
        if strict:
            target = sympify(target, locals=local_sympify)

            cond_iter = iterable(iter)
            cond_iter = cond_iter or isinstance(iter, (Range, Product,
                    Enumerate, Zip, Map))
            cond_iter = cond_iter or isinstance(iter, Variable) \
                and is_iterable_datatype(iter.dtype)
          #  cond_iter = cond_iter or isinstance(iter, ConstructorCall) \
          #      and is_iterable_datatype(iter.arguments[0].dtype)
            if not cond_iter:
                raise TypeError('iter must be an iterable')

            if iterable(body):
                body = CodeBlock((sympify(i, locals=local_sympify) for i in
                             body))
            elif not isinstance(body,CodeBlock):
                raise TypeError('body must be an iterable or a Codeblock')

        return Basic.__new__(cls, target, iter, body, local_vars)

    @property
    def target(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def local_vars(self):
        return self._args[3]

    def insert2body(self, stmt):
        self.body.append(stmt)



class DoConcurrent(For):
    pass


class ForAll(Basic):
    """ class that represents the forall statement in fortran"""
    def __new__(cls, iter, target, mask, body):

        if not isinstance(iter, Range):
            raise TypeError('iter must be of type Range')

        return Basic.__new__(cls, iter, target, mask, body)


    @property
    def iter(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

    @property
    def mask(self):
        return self._args[2]

    @property
    def body(self):
        return self._args[3]



class ForIterator(For):

    """Class that describes iterable classes defined by the user."""

    def __new__(
        cls,
        target,
        iter,
        body,
        strict=True,
        ):

        if isinstance(iter, Symbol):
            iter = Range(Len(iter))
        return For.__new__(cls, target, iter, body, strict)

    # TODO uncomment later when we intriduce iterators
    # @property
    # def target(self):
    #    ts = super(ForIterator, self).target

    #    if not(len(ts) == self.depth):
    #        raise ValueError('wrong number of targets')

    #    return ts

    @property
    def depth(self):
        it = self.iterable
        if isinstance(it, Variable):
            if isinstance(it.dtype, NativeRange):
                return 1
            if isinstance(it.dtype, NativeTensor):

                # TODO must be computed

                return 2

            cls_base = it.cls_base
            if not cls_base:
                raise TypeError('cls_base undefined')

            methods = cls_base.methods_as_dict

            it_method = methods['__iter__']

            it_vars = []
            for stmt in it_method.body:
                if isinstance(stmt, Assign):
                    it_vars.append(stmt.lhs)

            n = len(set([str(var.name) for var in it_vars]))
            return n
        else:

            return 1

    @property
    def ranges(self):
        return get_iterable_ranges(self.iterable)


# The following are defined to be sympy approved nodes. If there is something
# smaller that could be used, that would be preferable. We only use them as
# tokens.

class Is(Basic):

    """Represents a is expression in the code.

    Examples

    >>> from pyccel.ast import Is
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> Is(x, Nil())
    Is(x, None)
    """

    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]


class IsNot(Basic):

    """Represents a is expression in the code.

    Examples

    >>> from pyccel.ast import IsNot
    >>> from pyccel.ast import Nil
    >>> from sympy.abc import x
    >>> IsNot(x, Nil())
    IsNot(x, None)
    """

    def __new__(cls, lhs, rhs):
        return Basic.__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]



class ConstructorCall(AtomicExpr):

    """
    It  serves as a constructor for undefined function classes.

    func: FunctionDef, str
        an instance of FunctionDef or function name

    arguments: list, tuple, None
        a list of arguments.

    kind: str
        'function' or 'procedure'. default value: 'function'
    """

    is_commutative = True

    # TODO improve

    def __new__(
        cls,
        func,
        arguments,
        cls_variable=None,
        kind='function',
        ):
        if not isinstance(func, (FunctionDef, Interface, str)):
            raise TypeError('Expecting func to be a FunctionDef or str')

        if isinstance(func, FunctionDef):
            kind = func.kind

        f_name = func.name

        obj = Basic.__new__(cls, f_name)
        obj._cls_variable = cls_variable

        obj._kind = kind
        obj._func = func
        obj._arguments = arguments

        return obj

    def _sympystr(self, printer):
        sstr = printer.doprint
        name = sstr(self.name)
        args = ''
        if not self.arguments is None:
            args = ', '.join(sstr(i) for i in self.arguments)
        return '{0}({1})'.format(name, args)

    @property
    def func(self):
        return self._func

    @property
    def kind(self):
        return self._kind

    @property
    def arguments(self):
        return self._arguments

    @property
    def cls_variable(self):
        return self._cls_variable

    @property
    def name(self):
        if isinstance(self.func, FunctionDef):
            return self.func.name
        else:
            return self.func



class Nil(Basic):

    """
    class for None object in the code.
    """

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr('None')


class Void(Basic):

    pass

class VoidFunction(Basic):
    #this class is used in order to eliminate certain atoms
    # in an arithmitic expression so that we dont take them into
    # consideration
    def __new__(*args):
        return Symbol("""x9846548484665
                      494794564465165161561""")

class Variable(Symbol):

    """Represents a typed variable.

    dtype : str, DataType
        The type of the variable. Can be either a DataType,
        or a str (bool, int, real).

    name : str, list, DottedName
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
    >>> from pyccel.ast.core import Variable
    >>> Variable('int', 'n')
    n
    >>> Variable('real', x, rank=2, shape=(n,2), allocatable=True)
    x
    >>> Variable('int', ('matrix', 'n_rows'))
    matrix.n_rows
    """

    def __new__(
        cls,
        dtype,
        name,
        rank=0,
        allocatable=False,
        is_stack_array = False,
        is_pointer=False,
        is_target=False,
        is_polymorphic=None,
        is_optional=None,
        shape=None,
        cls_base=None,
        cls_parameters=None,
        order='C',
        precision=0
        ):

        if isinstance(dtype, str) or str(dtype) == '*':

            dtype = datatype(str(dtype))
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if allocatable is None:
            allocatable = False
        elif not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')

        if is_pointer is None:
            is_pointer = False
        elif not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')

        if is_target is None:
            is_target = False
        elif not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')

        if is_stack_array is None:
            is_stack_array = False
        elif not isinstance(is_stack_array, bool):
            raise TypeError('is_stack_array must be a boolean.')

        if is_polymorphic is None:
            if isinstance(dtype, CustomDataType):
                is_polymorphic = dtype.is_polymorphic
            else:
                is_polymorphic = False
        elif not isinstance(is_polymorphic, bool):
            raise TypeError('is_polymorphic must be a boolean.')

        if is_optional is None:
            is_optional = False
        elif not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')

        if not isinstance(precision,int):
            raise TypeError('precision must be an integer.')

        # if class attribut

        if isinstance(name, str):
            name = name.split(""".""")
            if len(name) == 1:
                name = name[0]
            else:
                name = DottedName(*name)

        if not isinstance(name, (str, DottedName)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(name)))

        if not isinstance(rank, int):
            raise TypeError('rank must be an instance of int.')

        if rank == 0:
            shape = ()

        if not precision:
            if isinstance(dtype, NativeInteger):
                precision = default_precision['int']
            elif isinstance(dtype, NativeReal):
                precision = default_precision['real']
            elif isinstance(dtype, NativeComplex):
                precision = default_precision['complex']
            elif isinstance(dtype, NativeBool):
                precision = default_precision['bool']

        # TODO improve order of arguments

        obj = Basic.__new__(
            cls,
            dtype,
            name,
            rank,
            allocatable,
            shape,
            cls_base,
            cls_parameters,
            is_pointer,
            is_target,
            is_polymorphic,
            is_optional,
            order,
            precision,
            is_stack_array,
            )

        assumptions = {}
        class_type = cls_base \
            or dtype.__class__.__name__.startswith('Pyccel')
        alloweddtypes = (NativeBool, NativeRange, NativeString,
                         NativeSymbol, NativeGeneric)

        if isinstance(dtype, NativeInteger):
            assumptions['integer'] = True
        elif isinstance(dtype, NativeReal):
            assumptions['real'] = True
        elif isinstance(dtype, NativeComplex):
            assumptions['complex'] = True
        elif isinstance(dtype, NativeBool):
            obj.is_Boolean = True
        elif not isinstance(dtype, alloweddtypes) and not class_type:
            raise TypeError('Undefined datatype')

        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy
        return obj

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

    @property
    def cls_parameters(self):
        return self._args[6]

    @property
    def is_pointer(self):
        return self._args[7]

    @property
    def is_target(self):
        return self._args[8]

    @property
    def is_polymorphic(self):
        return self._args[9]

    @property
    def is_optional(self):
        return self._args[10]

    @property
    def order(self):
        return self._args[11]

    @property
    def precision(self):
        return self._args[12]

    @property
    def is_stack_array(self):
        return self._args[13]

    @property
    def is_ndarray(self):
        """user friendly method to check if the variable is an ndarray:
            1. have a rank > 0
            2. dtype is one among {int, bool, real, complex}
        """

        if self.rank == 0:
            return False
        return isinstance(self.dtype, (NativeInteger, NativeBool,
                          NativeReal, NativeComplex))

    def __str__(self):
        if isinstance(self.name, (str, DottedName)):
            return str(self.name)
        elif self.name is iterable:
            return """.""".join(str(n) for n in self.name)

    def _sympystr(self, printer):
        sstr = printer.doprint
        if isinstance(self.name, (str, DottedName)):
            return '{}'.format(sstr(self.name))
        elif self.name is iterable:
            return """.""".join(sstr(n) for n in self.name)

    def inspect(self):
        """inspects the variable."""

        print('>>> Variable')
        print( '  name           = {}'.format(self.name))
        print( '  dtype          = {}'.format(self.dtype))
        print( '  rank           = {}'.format(self.rank))
        print( '  allocatable    = {}'.format(self.allocatable))
        print( '  shape          = {}'.format(self.shape))
        print( '  cls_base       = {}'.format(self.cls_base))
        print( '  cls_parameters = {}'.format(self.cls_parameters))
        print( '  is_pointer     = {}'.format(self.is_pointer))
        print( '  is_target      = {}'.format(self.is_target))
        print( '  is_polymorphic = {}'.format(self.is_polymorphic))
        print( '  is_optional    = {}'.format(self.is_optional))
        print( '<<<')

    def clone(self, name, new_class = None, **kwargs):

        # TODO check it is up to date

        if (new_class is None):
            cls = eval(self.__class__.__name__)
        else:
            cls = new_class

        return cls(
            self.dtype,
            name,
            rank=kwargs.pop('rank',self.rank),
            allocatable=kwargs.pop('allocatable',self.allocatable),
            shape=kwargs.pop('shape',self.shape),
            is_pointer=kwargs.pop('is_pointer',self.is_pointer),
            is_target=kwargs.pop('is_target',self.is_target),
            is_polymorphic=kwargs.pop('is_polymorphic',self.is_polymorphic),
            is_optional=kwargs.pop('is_optional',self.is_optional),
            cls_base=kwargs.pop('cls_base',self.cls_base),
            cls_parameters=kwargs.pop('cls_parameters',self.cls_parameters),
            )

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (
            self.dtype,
            self.name,
            self.rank,
            self.allocatable,
            self.is_pointer,
            self.is_polymorphic,
            self.is_optional,
            self.shape,
            self.cls_base,
            self.cls_parameters,
            )
        return args

    def _eval_subs(self, old, new):
        return self

    def _eval_is_positive(self):
        #we do this inorder to infere the type of Pow expression correctly
        return self.is_real


class DottedVariable(AtomicExpr, Boolean):

    """
    Represents a dotted variable.
    """

    def __new__(cls, *args):

        if not isinstance(args[0], (
            Variable,
            Symbol,
            IndexedVariable,
            IndexedElement,
            IndexedBase,
            Indexed,
            Function,
            DottedVariable,
            )):
            raise TypeError('Expecting a Variable or a function call, got instead {0} of type {1}'.format(str(args[0]),
                            type(args[0])))

        if not isinstance(args[1], (
            Variable,
            Symbol,
            IndexedVariable,
            IndexedElement,
            IndexedBase,
            Indexed,
            Function,
            )):
            raise TypeError('Expecting a Variable or a function call, got instead {0} of type {1}'.format(str(args[1]),
                            type(args[1])))

        obj = Basic.__new__(cls, args[0], args[1])
        assumptions = {}

        if args[1].is_integer:
            assumptions['integer'] = True
        elif args[1].is_real:
            assumptions['real'] = True
        elif args[1].is_complex:
            assumptions['complex'] = True

        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy
        return obj

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def rank(self):
        return self._args[1].rank

    @property
    def dtype(self):
        return self._args[1].dtype

    @property
    def allocatable(self):
        return self._args[1].allocatable

    @property
    def is_pointer(self):
        return self._args[1].is_pointer

    @property
    def is_target(self):
        return self._args[1].is_target

    @property
    def name(self):
        if isinstance(self.lhs, DottedVariable):
            name_0 = self.lhs.name
        else:
            name_0 = str(self.lhs)
        if isinstance(self.rhs, Function):
            name_1 = str(type(self.rhs).__name__)
        elif isinstance(self.rhs, Symbol):
            name_1 = self.rhs.name
        else:
            name_1 = str(self.rhs)
        return name_0 + """.""" + name_1

    def __str__(self):
        return self.name

    def _sympystr(self, Printer):
        return self.name

    @property
    def cls_base(self):
        return self._args[1].cls_base

    @property
    def names(self):
        """Return list of names as strings."""

        ls = []
        for i in [self.lhs, self.rhs]:
            if not isinstance(i, DottedVariable):
                ls.append(str(i))
            else:
                ls += i.names
        return ls

    def _eval_subs(self, old, new):
        return self

    def inspect(self):
        self._args[1].inspect()



class ValuedVariable(Variable):

    """Represents a valued variable in the code.

    variable: Variable
        A single variable
    value: Variable, or instance of Native types
        value associated to the variable

    Examples

    >>> from pyccel.ast.core import ValuedVariable
    >>> n  = ValuedVariable('int', 'n', value=4)
    >>> n
    n := 4
    """

    def __new__(cls, *args, **kwargs):

        # if value is not given, we set it to Nil
        # we also remove value from kwargs,
        # since it is not a valid argument for Variable

        value = kwargs.pop('value', Nil())

        obj = Variable.__new__(cls, *args, **kwargs)

        obj._value = value

        return obj

    @property
    def value(self):
        return self._value

    def _sympystr(self, printer):
        sstr = printer.doprint

        name = sstr(self.name)
        value = sstr(self.value)
        return '{0}={1}'.format(name, value)


class Constant(ValuedVariable):

    """

    Examples

    """

    pass


class Argument(Symbol):

    """An abstract Argument data structure.

    Examples

    >>> from pyccel.ast.core import Argument
    >>> n = Argument('n')
    >>> n
    n
    """

    pass


class ValuedArgument(Basic):

    """Represents a valued argument in the code.

    Examples

    >>> from pyccel.ast.core import ValuedArgument
    >>> n = ValuedArgument('n', 4)
    >>> n
    n=4
    """

    def __new__(cls, expr, value):
        if isinstance(expr, str):
            expr = Symbol(expr)

        # TODO should we turn back to Argument

        if not isinstance(expr, Symbol):
            raise TypeError('Expecting an argument')

        return Basic.__new__(cls, expr, value)

    @property
    def argument(self):
        return self._args[0]

    @property
    def value(self):
        return self._args[1]

    @property
    def name(self):
        return self.argument.name

    def _sympystr(self, printer):
        sstr = printer.doprint

        argument = sstr(self.argument)
        value = sstr(self.value)
        return '{0}={1}'.format(argument, value)


class FunctionCall(Basic):

    """Represents a function call in the code.
    """

    def __new__(cls, func, args):

        # ...
        if not isinstance(func, (str, FunctionDef, Function)):
            raise TypeError('> expecting a str, FunctionDef, Function')

        funcdef = None
        if isinstance(func, FunctionDef):
            funcdef = func
            func = func.name
        # ...

        # ...
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('> expecting an iterable')

        args = Tuple(*args, sympify=False)
        # ...

        obj = Basic.__new__(cls, func, args)

        obj._funcdef = funcdef

        return obj

    @property
    def func(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def funcdef(self):
        return self._funcdef

class Return(Basic):

    """Represents a function return in the code.

    expr : sympy expr
        The expression to return.

    stmts :represent assign stmts in the case of expression return
    """

    def __new__(cls, expr, stmt=None):

        if stmt and not isinstance(stmt, (Assign, CodeBlock)):
            raise TypeError('stmt should only be of type Assign')

        return Basic.__new__(cls, expr, stmt)

    @property
    def expr(self):
        return self._args[0]

    @property
    def stmt(self):
        return self._args[1]

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.expr, self.stmt)
        return args


class Interface(Basic):

    """Represent an Interface"""

    def __new__(
        cls,
        name,
        functions,
        hide=False,
        ):
        if not isinstance(name, str):
            raise TypeError('Expecting an str')
        if not isinstance(functions, list):
            raise TypeError('Expecting a list')
        return Basic.__new__(cls, name, functions, hide)

    @property
    def name(self):
        return self._args[0]

    @property
    def functions(self):
        return self._args[1]

    @property
    def hide(self):
        return self.functions[0].hide or self._args[2]

    @property
    def global_vars(self):
        return self.functions[0].global_vars

    @property
    def cls_name(self):
        return self.functions[0].cls_name

    @property
    def kind(self):
        return self.functions[0].kind

    @property
    def imports(self):
        return self.functions[0].imports

    @property
    def decorators(self):
        return self.functions[0].decorators

    @property
    def is_procedure(self):
        return self.functions[0].is_procedure

    def rename(self, newname):
        return Interface(newname, self.functions)


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

    is_pure: bool
        True for a function without side effect

    is_elemental: bool
        True for a function is elemental

    is_private: bool
        True for a function is private

    is_static: bool
        True for static functions. Needed for f2py

    is_external: bool
        True for a function will be visible with f2py

    is_external_call: bool
        True for a function call will be visible with f2py

    imports: list, tuple
        a list of needed imports

    decorators: list, tuple
        a list of proporties

    Examples

    >>> from pyccel.ast.core import Assign, Variable, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x]
    >>> results     = [y]
    >>> body        = [Assign(y,x+1)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x,), (y,), [y := 1 + x], [], [], None, False, function)

    One can also use parametrized argument, using ValuedArgument

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.core import FunctionDef
    >>> from pyccel.ast.core import ValuedArgument
    >>> from pyccel.ast.core import GetDefaultFunctionArg
    >>> n = ValuedArgument('n', 4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x, n=4), (y,), [y := 1 + x], [], [], None, False, function, [])
    """

    def __new__(
        cls,
        name,
        arguments,
        results,
        body,
        local_vars=[],
        global_vars=[],
        cls_name=None,
        hide=False,
        kind='function',
        is_static=False,
        imports=[],
        decorators={},
        header=None,
        is_recursive=False,
        is_pure=False,
        is_elemental=False,
        is_private=False,
        is_header=False,
        is_external=False,
        is_external_call=False,
        arguments_inout=[],
        functions = []):

        # name

        if isinstance(name, str):
            name = Symbol(name)
        elif isinstance(name, (tuple, list)):
            name_ = []
            for i in name:
                if isinstance(i, str):
                    name = name + Symbol(i)
                elif not isinstance(i, Symbol):
                    raise TypeError('Function name must be Symbol or string'
                                    )
            name = tuple(name_)
        elif not isinstance(name, Symbol):

            raise TypeError('Function name must be Symbol or string')

        # arguments

        if not iterable(arguments):
            raise TypeError('arguments must be an iterable')

        # TODO improve and uncomment
#        if not all(isinstance(a, Argument) for a in arguments):
#            raise TypeError("All arguments must be of type Argument")

        arguments = Tuple(*arguments, sympify=False)

        # body

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')

#        body = Tuple(*(i for i in body))
        # results

        if not iterable(results):
            raise TypeError('results must be an iterable')
        results = Tuple(*results, sympify=False)

        # if method

        if cls_name:

            if not isinstance(cls_name, str):
                raise TypeError('cls_name must be a string')

            # if not cls_variable:
             #   raise TypeError('Expecting a instance of {0}'.format(cls_name))

        if kind is None:
            kind = 'function'

        if not isinstance(kind, str):
            raise TypeError('Expecting a string for kind.')

        if not isinstance(is_static, bool):
            raise TypeError('Expecting a boolean for is_static attribut'
                            )

        if not kind in ['function', 'procedure']:
            raise ValueError("kind must be one among {'function', 'procedure'}"
                             )

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not isinstance(decorators, dict):
            raise TypeError('decorators must be a dict')

        if not isinstance(is_pure, bool):
            raise TypeError('Expecting a boolean for pure')

        if not isinstance(is_elemental, bool):
            raise TypeError('Expecting a boolean for elemental')

        if not isinstance(is_private, bool):
            raise TypeError('Expecting a boolean for private')

        if not isinstance(is_header, bool):
            raise TypeError('Expecting a boolean for private')


        if not isinstance(is_external, bool):
            raise TypeError('Expecting a boolean for external')

        if not isinstance(is_external_call, bool):
            raise TypeError('Expecting a boolean for external_call')

        if arguments_inout:
            if not isinstance(arguments_inout, (list, tuple, Tuple)):
                raise TypeError('Expecting an iterable ')

            if not all([isinstance(i, bool) for i in arguments_inout]):
                raise ValueError('Expecting booleans')

        else:
            # TODO shall we keep this?
            arguments_inout = [False for a in arguments]

        if functions:
            for i in functions:
                if not isinstance(i, FunctionDef):
                    raise TypeError('Expecting a FunctionDef')

        return Basic.__new__(
            cls,
            name,
            arguments,
            results,
            body,
            local_vars,
            global_vars,
            cls_name,
            hide,
            kind,
            is_static,
            imports,
            decorators,
            header,
            is_recursive,
            is_pure,
            is_elemental,
            is_private,
            is_header,
            is_external,
            is_external_call,
            arguments_inout,
            functions,)

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

    @property
    def is_static(self):
        return self._args[9]

    @property
    def imports(self):
        return self._args[10]

    @property
    def decorators(self):
        return self._args[11]

    @property
    def header(self):
        return self._args[12]

    @property
    def is_recursive(self):
        return self._args[13]

    @property
    def is_pure(self):
        return self._args[14]

    @property
    def is_elemental(self):
        return self._args[15]

    @property
    def is_private(self):
        return self._args[16]

    @property
    def is_header(self):
        return self._args[17]

    @property
    def is_external(self):
        return self._args[18]

    @property
    def is_external_call(self):
        return self._args[19]

    @property
    def arguments_inout(self):
        return self._args[20]

    @property
    def functions(self):
        return self._args[21]

    def print_body(self):
        for s in self.body:
            print(s)

    # TODO is there a better way to do this, avoiding copying args? => bad for
    # maintenance!
    #      must be done everywhere
    def set_recursive(self):
        return FunctionDef(
            self.name,
            self.arguments,
            self.results,
            self.body,
            local_vars=self.local_vars,
            global_vars=self.global_vars,
            cls_name=self.cls_name,
            hide=self.hide,
            kind=self.kind,
            is_static=self.is_static,
            header=self.header,
            imports = self.imports,
            decorators = self.decorators,
            is_recursive=True,
            functions=self.functions,
            )

    def rename(self, newname):
        """
        Rename the FunctionDef name by creating a new FunctionDef with
        newname.

        newname: str
            new name for the FunctionDef
        """

        return FunctionDef(
            newname,
            self.arguments,
            self.results,
            self.body,
            local_vars=self.local_vars,
            global_vars=self.global_vars,
            cls_name=self.cls_name,
            hide=self.hide,
            kind=self.kind,
            is_static=self.is_static,
            header=self.header,
            imports = self.imports,
            decorators = self.decorators,
            is_recursive=self.is_recursive,
            functions=self.functions,)

    def vectorize(self, body , header):
        """ return vectorized FunctionDef """
        decorators = self.decorators
        decorators.pop('vectorize')
        return FunctionDef(
            'vec_'+str(self.name),
            self.arguments,
            [],
            body,
            local_vars=self.local_vars,
            global_vars=self.global_vars,
            cls_name=self.cls_name,
            hide=self.hide,
            kind='procedure',
            is_static=self.is_static,
            header=header,
            imports = self.imports,
            decorators = decorators,
            is_recursive=self.is_recursive)

    @property
    def is_procedure(self):
        """Returns True if a procedure."""

        flag = False
        if len(self.results) == 1 and isinstance(self.results[0], Expr):
            vars_ = [i for i in preorder_traversal(self.results)
                     if isinstance(i, Variable)]
            flag = flag or any([i.allocatable or i.rank > 0 for i in
                               vars_])
        else:
            flag = flag or len(self.results) == 1 \
                and self.results[0].allocatable
            flag = flag or len(self.results) == 1 \
                and self.results[0].rank > 0
        flag = flag or len(self.results) > 1
        flag = flag or len(self.results) == 0
        flag = flag or self.kind == 'procedure'
        flag = flag \
            or len(set(self.results).intersection(self.arguments)) > 0
        return flag

    def is_compatible_header(self, header):
        """
        Returns True if the header is compatible with the given FunctionDef.

        header: Header
            a pyccel header suppose to describe the FunctionDef
        """

        cond_args = len(self.arguments) == len(header.dtypes)
        cond_results = len(self.results) == len(header.results)

        header_with_results = len(header.results) > 0

        if not cond_args:
            return False

        if header_with_results and not cond_results:
            return False

        return True

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (
            self.name,
            self.arguments,
            self.results,
            self.body,
            self.local_vars,
            self.global_vars,
            self.cls_name,
            self.hide,
            self.kind,
            self.is_static,
            self.imports,
            self.decorators,
            self.header,
            self.is_recursive,
            )
        return args

    # TODO
    def check_pure(self):
        raise NotImplementedError('')

    # TODO
    def check_elemental(self):
        raise NotImplementedError('')


class SympyFunction(FunctionDef):

    """Represents a function definition."""

    def rename(self, newname):
        """
        Rename the SympyFunction name by creating a new SympyFunction with
        newname.

        newname: str
            new name for the SympyFunction
        """

        return SympyFunction(newname, self.arguments, self.results,
                             self.body, cls_name=self.cls_name)


class PythonFunction(FunctionDef):

    """Represents a Python-Function definition."""

    def rename(self, newname):
        """
        Rename the PythonFunction name by creating a new PythonFunction with
        newname.

        newname: str
            new name for the PythonFunction
        """

        return PythonFunction(newname, self.arguments, self.results,
                              self.body, cls_name=self.cls_name)


class F2PYFunctionDef(FunctionDef):
    pass


class GetDefaultFunctionArg(Basic):

    """Creates a FunctionDef for handling optional arguments in the code.

    arg: ValuedArgument, ValuedVariable
        argument for which we want to create the function returning the default
        value

    func: FunctionDef
        the function/subroutine in which the optional arg is used

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.core import FunctionDef
    >>> from pyccel.ast.core import ValuedArgument
    >>> from pyccel.ast.core import GetDefaultFunctionArg
    >>> n = ValuedArgument('n', 4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> incr = FunctionDef('incr', args, results, body)
    >>> get_n = GetDefaultFunctionArg(n, incr)
    >>> get_n.name
    get_default_incr_n
    >>> get_n
    get_default_incr_n(n=4)

    You can also use **ValuedVariable** as in the following example

    >>> from pyccel.ast.core import ValuedVariable
    >>> n = ValuedVariable('int', 'n', value=4)
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> incr = FunctionDef('incr', args, results, body)
    >>> get_n = GetDefaultFunctionArg(n, incr)
    >>> get_n
    get_default_incr_n(n=4)
    """

    def __new__(cls, arg, func):

        if not isinstance(arg, (ValuedArgument, ValuedVariable)):
            raise TypeError('Expecting a ValuedArgument or ValuedVariable'
                            )

        if not isinstance(func, FunctionDef):
            raise TypeError('Expecting a FunctionDef')

        return Basic.__new__(cls, arg, func)

    @property
    def argument(self):
        return self._args[0]

    @property
    def func(self):
        return self._args[1]

    @property
    def name(self):
        text = \
            'get_default_{func}_{arg}'.format(arg=self.argument.name,
                func=self.func.name)
        return text

    def _sympystr(self, printer):
        sstr = printer.doprint

        name = sstr(self.name)
        argument = sstr(self.argument)
        return '{0}({1})'.format(name, argument)


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

    imports: list, tuple
        list of needed imports

    parent : str
        parent's class name

    Examples

    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef
    >>> x = Variable('real', 'x')
    >>> y = Variable('real', 'y')
    >>> z = Variable('real', 'z')
    >>> t = Variable('real', 't')
    >>> a = Variable('real', 'a')
    >>> b = Variable('real', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributs   = [x,y]
    >>> methods     = [translate]
    >>> ClassDef('Point', attributs, methods)
    ClassDef(Point, (x, y), (FunctionDef(translate, (x, y, a, b), (z, t), [y := a + x], [], [], None, False, function),), [public])
    """

    def __new__(
        cls,
        name,
        attributes=[],
        methods=[],
        options=['public'],
        imports=[],
        parent=[],
        interfaces=[],
        ):

        # name

        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, Symbol):
            raise TypeError('Function name must be Symbol or string')

        # attributs

        if not iterable(attributes):
            raise TypeError('attributs must be an iterable')
        attributes = Tuple(*attributes, sympify=False)

        # methods

        if not iterable(methods):
            raise TypeError('methods must be an iterable')

        # options

        if not iterable(options):
            raise TypeError('options must be an iterable')

        # imports

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not iterable(parent):
            raise TypeError('parent must be iterable')

        if not iterable(interfaces):
            raise TypeError('interfaces must be iterable')

        imports = list(imports)
        for i in methods:
            imports += list(i.imports)

        imports = set(imports)  # for unicity
        imports = Tuple(*imports, sympify=False)

        # ...
        # look if the class has the method __del__
        # d_methods = {}
        # for i in methods:
        #    d_methods[str(i.name).replace('\'','')] = i
        # if not ('__del__' in d_methods):
        #    dtype = DataTypeFactory(str(name), ("_name"), prefix='Custom')
        #    this  = Variable(dtype(), 'self')

            # constructs the __del__ method if not provided
         #   args = []
         #   for a in attributs:
         #       if isinstance(a, Variable):
         #           if a.allocatable:
         #              args.append(a)

         #   args = [Variable(a.dtype, DottedName(str(this), str(a.name))) for a in args]
         #   body = [Del(a) for a in args]

         #   free = FunctionDef('__del__', [this], [], \
         #                      body, local_vars=[], global_vars=[], \
         #                      cls_name='__UNDEFINED__', kind='procedure', imports=[])

         #  methods = list(methods) + [free]
         # TODO move this somewhere else

        methods = Tuple(*methods, sympify=False)

        # ...

        return Basic.__new__(
            cls,
            name,
            attributes,
            methods,
            options,
            imports,
            parent,
            interfaces,
            )

    @property
    def name(self):
        return self._args[0]

    @property
    def attributes(self):
        return self._args[1]

    @property
    def methods(self):
        return self._args[2]

    @property
    def options(self):
        return self._args[3]

    @property
    def imports(self):
        return self._args[4]

    @property
    def parent(self):
        return self._args[5]

    @property
    def interfaces(self):
        return self._args[6]

    @property
    def methods_as_dict(self):
        """Returns a dictionary that contains all methods, where the key is the
        method's name."""

        d_methods = {}
        for i in self.methods:
            d_methods[str(i.name)] = i
        return d_methods

    @property
    def attributes_as_dict(self):
        """Returns a dictionary that contains all attributs, where the key is the
        attribut's name."""

        d_attributes = {}
        for i in self.attributes:
            d_attributes[str(i.name)] = i
        return d_attributes

    # TODO add other attributes?


    def get_attribute(self, O, attr):
        """Returns the attribute attr of the class O of instance self."""

        if not isinstance(attr, str):
            raise TypeError('Expecting attribute to be a string')

        if isinstance(O, Variable):
            cls_name = str(O.name)
        else:
            cls_name = str(O)

        attributes = {}
        for i in self.attributes:
            attributes[str(i.name)] = i

        if not attr in attributes:
            raise ValueError('{0} is not an attribut of {1}'.format(attr,
                             str(self)))

        var = attributes[attr]
        name = DottedName(cls_name, str(var.name))
        return Variable(
            var.dtype,
            name,
            rank=var.rank,
            allocatable=var.allocatable,
            shape=var.shape,
            cls_base=var.cls_base,
            )

    @property
    def is_iterable(self):
        """Returns True if the class has an iterator."""

        names = [str(m.name) for m in self.methods]
        if '__next__' in names and '__iter__' in names:
            return True
        elif '__next__' in names:
            raise ValueError('ClassDef does not contain __iter__ method')
        elif '__iter__' in names:
            raise ValueError('ClassDef does not contain __next__ method')
        else:
            return False

    @property
    def is_with_construct(self):
        """Returns True if the class is a with construct."""

        names = [str(m.name) for m in self.methods]
        if '__enter__' in names and '__exit__' in names:
            return True
        elif '__enter__' in names:
            raise ValueError('ClassDef does not contain __exit__ method')
        elif '__exit__' in names:
            raise ValueError('ClassDef does not contain __enter__ method')
        else:
            return False

    @property
    def hide(self):
        if 'hide' in self.options:
            return True
        else:
            return self.is_iterable or self.is_with_construct

    def _eval_subs(self, old , new):
        return self


class Import(Basic):

    """Represents inclusion of dependencies in the code.

    target : str, list, tuple, Tuple
        targets to import

    Examples

    >>> from pyccel.ast.core import Import
    >>> from pyccel.ast.core import DottedName
    >>> Import('foo')
    import foo

    >>> abc = DottedName('foo', 'bar', 'baz')
    >>> Import(abc)
    import foo.bar.baz

    >>> Import(['foo', abc])
    import foo, foo.bar.baz
    """

    def __new__(cls, target, source=None):

        def _format(i):
            if isinstance(i, str):
                if '.' in i:
                    return DottedName(*i.split('.'))
                else:
                    return Symbol(i)
            if isinstance(i, (DottedName, AsName)):
                return i
            elif isinstance(i, Symbol):
                return i
            else:
                raise TypeError('Expecting a string, Symbol DottedName, given {}'.format(type(i)))

        _target = []
        if isinstance(target, (str, Symbol, DottedName, AsName)):
            _target = [_format(target)]
        elif iterable(target):
            for i in target:
                _target.append(_format(i))
        target = Tuple(*_target, sympify=False)

        if not source is None:
            source = _format(source)

        return Basic.__new__(cls, target, source)

    @property
    def target(self):
        return self._args[0]

    @property
    def source(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        target = ', '.join([sstr(i) for i in self.target])
        if self.source is None:
            return 'import {target}'.format(target=target)
        else:
            source = sstr(self.source)
            return 'from {source} import {target}'.format(source=source,
                    target=target)


class TupleImport(Basic):

    def __new__(cls, *args):
        for a in args:
            if not isinstance(a, Import):
                raise TypeError('Expecting an Import statement')
        return Basic.__new__(cls, *args)

    @property
    def imports(self):
        return self._args

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '\n'.join(sstr(n) for n in self.imports)


class Load(Basic):

    """Similar to 'importlib' in python. In addition, we can also provide the
    functions we want to import.

    module: str, DottedName
        name of the module to load.

    funcs: str, list, tuple, Tuple
        a string representing the function to load, or a list of strings.

    as_lambda: bool
        load as a Lambda expression, if True

    nargs: int
        number of arguments of the function to load. (default = 1)

    Examples

    >>> from pyccel.ast.core import Load
    """

    def __new__(
        cls,
        module,
        funcs=None,
        as_lambda=False,
        nargs=1,
        ):
        if not isinstance(module, (str, DottedName, list, tuple,
                          Tuple)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(module)))

        # see syntax

        if isinstance(module, str):
            module = module.replace('__', """.""")

        if isinstance(module, (list, tuple, Tuple)):
            module = DottedName(*module)

        if funcs:
            if not isinstance(funcs, (str, DottedName, list, tuple,
                              Tuple)):
                raise TypeError('Expecting a string or DottedName')

            if isinstance(funcs, str):
                funcs = [funcs]
            elif not isinstance(funcs, (list, tuple, Tuple)):
                raise TypeError('Expecting a string, list, tuple, Tuple')

        if not isinstance(as_lambda, (BooleanTrue, BooleanFalse, bool)):
            raise TypeError('Expecting a boolean, given {0}'.format(as_lambda))

        return Basic.__new__(cls, module, funcs, as_lambda, nargs)

    @property
    def module(self):
        return self._args[0]

    @property
    def funcs(self):
        return self._args[1]

    @property
    def as_lambda(self):
        return self._args[2]

    @property
    def nargs(self):
        return self._args[3]

    def execute(self):
        module = str(self.module)
        try:
            package = importlib.import_module(module)
        except:
            raise ImportError('could not import {0}'.format(module))

        ls = []
        for f in self.funcs:
            try:
                m = getattr(package, '{0}'.format(str(f)))
            except:
                raise ImportError('could not import {0}'.format(f))

            # TODO improve

            if self.as_lambda:
                args = []
                for i in range(0, self.nargs):
                    fi = Symbol('f{0}'.format(i))
                    args.append(fi)
                if len(args) == 1:
                    arg = args[0]
                    m = Lambda(arg, m(arg, evaluate=False))
                else:
                    m = Lambda(args, m(evaluate=False, *args))

            ls.append(m)

        return ls


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
    value: Expr
        variable value
    static: bool
        True for a static declaration of an array.

    Examples

    >>> from pyccel.ast.core import Declare, Variable
    >>> Declare('int', Variable('int', 'n'))
    Declare(NativeInteger(), (n,), None)
    >>> Declare('real', Variable('real', 'x'), intent='out')
    Declare(NativeReal(), (x,), out)
    """

    def __new__(
        cls,
        dtype,
        variable,
        intent=None,
        value=None,
        static=False,
        ):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(variable, Variable):
            raise TypeError('var must be of type Variable, given {0}'.format(variable))
        if variable.dtype != dtype:
            raise ValueError('All variables must have the same dtype')

        if intent:
            if not intent in ['in', 'out', 'inout']:
                raise ValueError("intent must be one among {'in', 'out', 'inout'}")

        if not isinstance(static, bool):
            raise TypeError('Expecting a boolean for static attribut')

        return Basic.__new__(
            cls,
            dtype,
            variable,
            intent,
            value,
            static,
            )

    @property
    def dtype(self):
        return self._args[0]

    @property
    def variable(self):
        return self._args[1]

    @property
    def intent(self):
        return self._args[2]

    @property
    def value(self):
        return self._args[3]

    @property
    def static(self):
        return self._args[4]


class Subroutine(UndefinedFunction):

    pass


class Break(Basic):

    """Represents a break in the code."""

    pass


class Continue(Basic):

    """Represents a continue in the code."""

    pass


class Raise(Basic):

    """Represents a raise in the code."""

    pass


# TODO: improve with __new__ from Function and add example

class Random(Function):

    """
    Represents a 'random' number in the code.
    """

    # TODO : remove later

    def __str__(self):
        return 'random'

    def __new__(cls, seed):
        return Basic.__new__(cls, seed)

    @property
    def seed(self):
        return self._args[0]


# TODO: improve with __new__ from Function and add example

class SumFunction(Basic):

    """Represents a Sympy Sum Function.

       body: Expr
       Sympy Expr in which the sum will be performed.

       iterator:
       a tuple  that containts the index of the sum and it's range.
    """

    def __new__(
        cls,
        body,
        iterator,
        stmts=None,
        ):
        if not isinstance(iterator, (tuple, Tuple)):
            raise TypeError('iterator must be a tuple')
        if not len(iterator) == 3:
            raise ValueError('iterator must be of lenght 3')
        return Basic.__new__(cls, body, iterator, stmts)

    @property
    def body(self):
        return self._args[0]

    @property
    def iterator(self):
        return self._args[1]

    @property
    def stmts(self):
        return self._args[2]


class SymbolicPrint(Basic):

    """Represents a print function of symbolic expressions in the code.

    expr : sympy expr
        The expression to return.

    Examples

    >>> from sympy import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """

    def __new__(cls, expr):
        if not iterable(expr):
            raise TypeError('Expecting an iterable')

        for i in expr:
            if not isinstance(i, (Lambda, SymbolicAssign,
                              SympyFunction)):
                raise TypeError('Expecting Lambda, SymbolicAssign, SympyFunction for {}'.format(i))

        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]


class Del(Basic):

    """Represents a memory deallocation in the code.

    variables : list, tuple
        a list of pyccel variables

    Examples

    >>> from pyccel.ast.core import Del, Variable
    >>> x = Variable('real', 'x', rank=2, shape=(10,2), allocatable=True)
    >>> Del([x])
    Del([x])
    """

    def __new__(cls, expr):

        # TODO: check that the variable is allocatable

        if not iterable(expr):
            expr = Tuple(expr, sympify=False)
        return Basic.__new__(cls, expr)

    @property
    def variables(self):
        return self._args[0]


class EmptyLine(Basic):

    """Represents a EmptyLine in the code.

    text : str
       the comment line

    Examples

    >>> from pyccel.ast.core import EmptyLine
    >>> EmptyLine()

    """

    def __new__(cls):
        return Basic.__new__(cls)

    def _sympystr(self, printer):
        return ''


class NewLine(Basic):

    """Represents a NewLine in the code.

    text : str
       the comment line

    Examples

    >>> from pyccel.ast.core import NewLine
    >>> NewLine()

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

    >>> from pyccel.ast.core import Comment
    >>> Comment('this is a comment')
    # this is a comment
    """

    def __new__(cls, text):
        return Basic.__new__(cls, text)

    @property
    def text(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '# {0}'.format(sstr(self.text))


class SeparatorComment(Comment):

    """Represents a Separator Comment in the code.

    mark : str
        marker

    Examples

    >>> from pyccel.ast.core import SeparatorComment
    >>> SeparatorComment(n=40)
    # ........................................
    """

    def __new__(cls, n):
        text = """.""" * n
        return Comment.__new__(cls, text)


class AnnotatedComment(Basic):

    """Represents a Annotated Comment in the code.

    accel : str
       accelerator id. One among {'omp', 'acc'}

    txt: str
        statement to print

    Examples

    >>> from pyccel.ast.core import AnnotatedComment
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

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.accel, self.txt)
        return args

class CommentBlock(Basic):

    """ Represents a Block of Comments
        txt : str
    """
    def __new__(cls, txt):
        if not isinstance(txt, str):
            raise TypeError('txt must be of type str')
        txt = txt.replace('"','')
        txts = txt.split('\n')

        return Basic.__new__(cls, txts)

    @property
    def comments(self):
        return self._args[0]

class IndexedVariable(IndexedBase):

    """
    Represents an indexed variable, like x in x[i], in the code.

    Examples

    >>> from sympy import symbols, Idx
    >>> from pyccel.ast.core import IndexedVariable
    >>> A = IndexedVariable('A'); A
    A
    >>> type(A)
    <class 'pyccel.ast.core.IndexedVariable'>

    When an IndexedVariable object receives indices, it returns an array with named
    axes, represented by an IndexedElement object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'pyccel.ast.core.IndexedElement'>

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

    def __new__(
        cls,
        label,
        shape=None,
        dtype=None,
        prec=0,
        order=None,
        rank = 0,
        **kw_args
        ):
        if dtype is None:
            raise TypeError('datatype must be provided')
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        obj = IndexedBase.__new__(cls, label, shape=shape)
        kw_args['dtype']     = dtype
        kw_args['precision'] = prec
        kw_args['order']     = order
        kw_args['rank']      = rank
        obj._kw_args         = kw_args

        return obj

    def __getitem__(self, *args):

        if self.shape and len(self.shape) != len(args):
            raise IndexException('Rank mismatch.')
        assumptions = {}
        obj = IndexedElement(self, *args)
        return obj

    @property
    def dtype(self):
        return self.kw_args['dtype']

    @property
    def precision(self):
        return self.kw_args['precision']

    @property
    def order(self):
        return self.kw_args['order']

    @property
    def rank(self):
        return self.kw_args['rank']

    @property
    def kw_args(self):
        return self._kw_args

    @property
    def name(self):
        return self._args[0]



    def clone(self, name):
        cls = eval(self.__class__.__name__)
        # TODO what about kw_args in __new__?
        return cls(name, shape=self.shape, dtype=self.dtype,
                   prec=self.precision, order=self.order, rank=self.rank)

    def _eval_subs(self, old, new):
        return self

    def __str__(self):
        return str(self.name)


class IndexedElement(Indexed):

    """
    Represents a mathematical object with indices.

    Examples

    >>> from sympy import symbols, Idx
    >>> from pyccel.ast.core import IndexedVariable
    >>> i, j = symbols('i j', cls=Idx)
    >>> IndexedElement('A', i, j)
    A[i, j]

    It is recommended that ``IndexedElement`` objects be created via ``IndexedVariable``:

    >>> from pyccel.ast.core import IndexedElement
    >>> A = IndexedVariable('A')
    >>> IndexedElement('A', i, j) == A[i, j]
    False

    **todo:** fix bug. the last result must be : True
    """

    def __new__(
        cls,
        base,
        *args,
        **kw_args
        ):

        if not args:
            raise IndexException('Indexed needs at least one index.')
        if isinstance(base, (string_types, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base,
                IndexedBase):
            raise TypeError(filldedent("""
                Indexed expects string, Symbol, or IndexedBase as base."""))

        args_ = []

        for arg in args:
            args_.append(sympify(arg, locals=local_sympify))
        args = args_

        if isinstance(base, (NDimArray, Iterable, Tuple,
                      MatrixBase)) and all([i.is_number for i in args]):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]
        obj = Expr.__new__(cls, base, *args, **kw_args)
        alloweddtypes = (NativeBool, NativeRange, NativeString)
        dtype = obj.base.dtype
        assumptions = {}
        if isinstance(dtype, NativeInteger):
            assumptions['integer'] = True
        elif isinstance(dtype, NativeReal):
            assumptions['real'] = True
        elif isinstance(dtype, NativeComplex):
            assumptions['complex'] = True
        elif not isinstance(dtype, alloweddtypes):
            raise TypeError('Undefined datatype')
        ass_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = ass_copy

        return obj

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
            if not isinstance(a, Slice):
                n += 1
        return n

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def precision(self):
        return self.base.precision

    @property
    def order(self):
        return self.base.order

    def _eval_subs(self, old, new):
        return self



class String(Basic):

    """Represents the String"""

    def __new__(cls, arg):
        if not isinstance(arg, str):
            raise TypeError('arg must be of type str')
        return Basic.__new__(cls, arg)

    @property
    def arg(self):
        return self._args[0]

    def __str__(self):
        return self.arg


class Concatenate(Basic):

    """Represents the String concatination operation.

    left : Symbol or string or List

    right : Symbol or string or List


    Examples

    >>> from sympy import symbols
    >>> from pyccel.ast.core import Concatenate
    >>> x = symbols('x')
    >>> Concatenate('some_string',x)
    some_string+x
    >>> Concatenate('some_string','another_string')
    'some_string' + 'another_string'
    """

    # TODO add step

    def __new__(cls, args, is_list):
        args = list(args)

        args = [ repr(arg) if isinstance(arg, str) else arg for arg in args]

        return Basic.__new__(cls, args, is_list)

    @property
    def args(self):
        return self._args[0]

    @property
    def is_list(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint

        args = '+'.join(sstr(arg) for arg in self.args)

        return args




class Slice(Basic):

    """Represents a slice in the code.

    start : Symbol or int
        starting index

    end : Symbol or int
        ending index

    Examples

    >>> from sympy import symbols
    >>> from pyccel.ast.core import Slice
    >>> m, n = symbols('m, n', integer=True)
    >>> Slice(m,n)
    m : n
    >>> Slice(None,n)
     : n
    >>> Slice(m,None)
    m :
    """

    # TODO add step
    # TODO check that args are integers
    # TODO add negative indices
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

    def __str__(self):
        if self.start is None:
            start = ''
        else:
            start = str(self.start)
        if self.end is None:
            end = ''
        else:
            end = str(self.end)
        return '{0} : {1}'.format(start, end)


class Assert(Basic):

    """Represents a assert statement in the code.

    test: Expr
        boolean expression to check

    Examples

    """

    def __new__(cls, test):
        if not isinstance(test, (bool, Relational, Boolean)):
            raise TypeError('test %s is of type %s, but must be a Relational, Boolean, or a built-in bool.'
                             % (test, type(test)))

        return Basic.__new__(cls, test)

    @property
    def test(self):
        return self._args[0]


class Eval(Basic):

    """Basic class for eval instruction."""

    pass


class Pass(Basic):

    """Basic class for pass instruction."""

    pass


class Exit(Basic):

    """Basic class for exists."""

    pass


class ErrorExit(Exit):

    """Exist with error."""

    pass


class If(Basic):

    """Represents a if statement in the code.

    args :
        every argument is a tuple and
        is defined as (cond, expr) where expr is a valid ast element
        and cond is a boolean test.

    Examples

    >>> from sympy import Symbol
    >>> from pyccel.ast.core import Assign, If
    >>> n = Symbol('n')
    >>> If(((n>1), [Assign(n,n-1)]), (True, [Assign(n,n+1)]))
    If(((n>1), [Assign(n,n-1)]), (True, [Assign(n,n+1)]))
    """

    # TODO add step

    def __new__(cls, *args):

        # (Try to) sympify args first

        newargs = []
        for ce in args:
            cond = ce[0]
            if not isinstance(cond, (bool, Relational, Boolean, Is, IsNot)):
                raise TypeError('Cond %s is of type %s, but must be a Relational, Boolean, Is, IsNot, or a built-in bool.'
                                 % (cond, type(cond)))
            if isinstance(ce[1], (list, Tuple, tuple)):
                body = CodeBlock(ce[1])
            elif isinstance(ce[1], CodeBlock):
                body = ce[1]
            else:
                raise TypeError('body is not iterable or CodeBlock')
            newargs.append((cond,body))

        return Basic.__new__(cls, *newargs)

    @property
    def bodies(self):
        b = []
        for i in self._args:
            b.append( i[1])
        return b


class IfTernaryOperator(If):

    """class for the Ternery operator"""

    pass


def is_simple_assign(expr):
    if not isinstance(expr, Assign):
        return False

    assignable = [Variable, IndexedVariable, IndexedElement]
    assignable += [sp_Integer, sp_Float]
    assignable = tuple(assignable)
    if isinstance(expr.rhs, assignable):
        return True
    else:
        return False


# ...

# ...

def get_initial_value(expr, var):
    """Returns the first assigned value to var in the Expression expr.

    expr: Expression
        any AST valid expression

    var: str, Variable, DottedName, list, tuple
        variable name
    """

    # ...

    def is_None(expr):
        """Returns True if expr is None or Nil()."""

        return isinstance(expr, Nil) or expr is None

    # ...

    # ...

    if isinstance(var, str):
        return get_initial_value(expr, [var])
    elif isinstance(var, DottedName):

        return get_initial_value(expr, [str(var)])
    elif isinstance(var, Variable):

        return get_initial_value(expr, [var.name])
    elif not isinstance(var, (list, tuple)):

        raise TypeError('Expecting var to be str, list, tuple or Variable, given {0}'.format(type(var)))

    # ...

    # ...

    if isinstance(expr, ValuedVariable):
        if expr.variable.name in var:
            return expr.value
    elif isinstance(expr, Variable):

        # expr.cls_base if of type ClassDef

        if expr.cls_base:
            return get_initial_value(expr.cls_base, var)
    elif isinstance(expr, Assign):

        if str(expr.lhs) in var:
            return expr.rhs
    elif isinstance(expr, FunctionDef):

        value = get_initial_value(expr.body, var)
        if not is_None(value):
            r = get_initial_value(expr.arguments, value)
            if 'self._linear' in var:
                print ('>>>> ', var, value, r)
            if not r is None:
                return r
        return value

    elif isinstance(expr, ConstructorCall):

        return get_initial_value(expr.func, var)
    elif isinstance(expr, (list, tuple, Tuple)):

        for i in expr:
            value = get_initial_value(i, var)

            # here we make a difference between None and Nil,
            # since the output of our function can be None

            if not value is None:
                return value
    elif isinstance(expr, ClassDef):

        methods = expr.methods_as_dict
        init_method = methods['__init__']
        return get_initial_value(init_method, var)

    # ...

    return Nil()


# ...

# ... TODO treat other statements

def get_assigned_symbols(expr):
    """Returns all assigned symbols (as sympy Symbol) in the AST.

    expr: Expression
        any AST valid expression
    """

    if isinstance(expr, (CodeBlock, FunctionDef, For, While)):
        return get_assigned_symbols(expr.body)
    elif isinstance(expr, FunctionalFor):
        return get_assigned_symbols(expr.loops)
    elif isinstance(expr, If):

        return get_assigned_symbols(expr.bodies)

    elif iterable(expr):
        symbols = []

        for a in expr:
            symbols += get_assigned_symbols(a)
        symbols = set(symbols)
        symbols = list(symbols)
        return symbols
    elif isinstance(expr, (Assign, AugAssign)):


        if expr.lhs is None:
            raise TypeError('Found None lhs')

        try:
            var = expr.lhs
            symbols = []
            if isinstance(var, DottedVariable):
                var = expr.lhs.lhs
                while isinstance(var, DottedVariable):
                    var = var.lhs
                symbols.append(var)
            elif isinstance(var, IndexedElement):
                var = var.base
                symbols.append(var)
            return symbols
        except:
            #TODO should we keep the try/except clause ?

            # TODO must raise an Exception here
            #      this occurs only when parsing lapack.pyh
            raise ValueError('Unable to extract assigned variable')
#            print(type(expr.lhs), expr.lhs)
#            print(expr)
#            raise SystemExit('ERROR')

    return []


# ...

# ... TODO: improve and make it recursive

def get_iterable_ranges(it, var_name=None):
    """Returns ranges of an iterable object."""

    if isinstance(it, Variable):
        if it.cls_base is None:
            raise TypeError('iterable must be an iterable Variable object'
                            )

        # ...

        def _construct_arg_Range(name):
            if not isinstance(name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(name)))

            if not var_name:
                return DottedName(it.name.name[0], name.name[1])
            else:
                return DottedName(var_name, name.name[1])

        # ...

        cls_base = it.cls_base

        if isinstance(cls_base, Range):
            if not isinstance(it.name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(it.name)))

            args = []
            for i in [cls_base.start, cls_base.stop, cls_base.step]:
                if isinstance(i, (Variable, IndexedVariable)):
                    arg_name = _construct_arg_Range(i.name)
                    arg = i.clone(arg_name)
                elif isinstance(i, IndexedElement):
                    arg_name = _construct_arg_Range(i.base.name)
                    base = i.base.clone(arg_name)
                    indices = i.indices
                    arg = base[indices]
                else:
                    raise TypeError('Wrong type, given {0}'.format(type(i)))
                args += [arg]

            return [Range(*args)]
        elif isinstance(cls_base, Tensor):

            if not isinstance(it.name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(it.name)))

            # ...

            ranges = []
            for r in cls_base.ranges:
                ranges += get_iterable_ranges(r,
                        var_name=str(it.name.name[0]))

            # ...

            return ranges

        params = [str(i) for i in it.cls_parameters]
    elif isinstance(it, ConstructorCall):
        cls_base = it.this.cls_base

        # arguments[0] is 'self'
        # TODO must be improved in syntax, so that a['value'] is a sympy object

        args = []
        kwargs = {}
        for a in it.arguments[1:]:
            if isinstance(a, dict):

                # we add '_' tp be conform with the private variables convention

                kwargs['{0}'.format(a['key'])] = a['value']
            else:
                args.append(a)

        # TODO improve

        params = args

#        for k,v in kwargs:
#            params.append(k)

    methods = cls_base.methods_as_dict
    init_method = methods['__init__']

    args = init_method.arguments[1:]
    args = [str(i) for i in args]

    # ...

    it_method = methods['__iter__']
    targets = []
    starts = []
    for stmt in it_method.body:
        if isinstance(stmt, Assign):
            targets.append(stmt.lhs)
            starts.append(stmt.lhs)

    names = []
    for i in starts:
        if isinstance(i, IndexedElement):
            names.append(str(i.base))
        else:
            names.append(str(i))
    names = list(set(names))

    inits = {}
    for stmt in init_method.body:
        if isinstance(stmt, Assign):
            if str(stmt.lhs) in names:
                expr = stmt.rhs
                for (a_old, a_new) in zip(args, params):
                    dtype = datatype(stmt.rhs.dtype)
                    v_old = Variable(dtype, a_old)
                    if isinstance(a_new, (IndexedVariable,
                                  IndexedElement, str, Variable)):
                        v_new = Variable(dtype, a_new)
                    else:
                        v_new = a_new
                    expr = subs(expr, v_old, v_new)
                    inits[str(stmt.lhs)] = expr

    _starts = []
    for i in starts:
        if isinstance(i, IndexedElement):
            _starts.append(i.base)
        else:
            _starts.append(i)
    starts = [inits[str(i)] for i in _starts]

    # ...

    def _find_stopping_criterium(stmts):
        for stmt in stmts:
            if isinstance(stmt, If):
                if not len(stmt.args) == 2:
                    raise ValueError('Wrong __next__ pattern')

                (ct, et) = stmt.args[0]
                (cf, ef) = stmt.args[1]

                for i in et:
                    if isinstance(i, Raise):
                        return cf

                for i in ef:
                    if isinstance(i, Raise):
                        return ct

                raise TypeError('Wrong type for __next__ pattern')

        return None

    # ...

    # ...

    def doit(expr, targets):
        if isinstance(expr, Relational):
            if str(expr.lhs) in targets and expr.rel_op in ['<', '<=']:
                return expr.rhs
            elif str(expr.rhs) in targets and expr.rel_op in ['>', '>='
                    ]:
                return expr.lhs
            else:
                return None
        elif isinstance(expr, And):
            return [doit(a, targets) for a in expr.args]
        else:
            raise TypeError('Expecting And logical expression.')

    # ...

    # ...

    next_method = methods['__next__']
    ends = []
    cond = _find_stopping_criterium(next_method.body)

    # TODO treate case of cond with 'and' operation
    # TODO we should avoid using str
    #      must change target from DottedName to Variable

    targets = [str(i) for i in targets]
    ends = doit(cond, targets)

    # TODO not use str

    if not isinstance(ends, (list, tuple)):
        ends = [ends]

    names = []
    for i in ends:
        if isinstance(i, IndexedElement):
            names.append(str(i.base))
        else:
            names.append(str(i))
    names = list(set(names))

    inits = {}
    for stmt in init_method.body:
        if isinstance(stmt, Assign):
            if str(stmt.lhs) in names:
                expr = stmt.rhs
                for (a_old, a_new) in zip(args, params):
                    dtype = datatype(stmt.rhs.dtype)
                    v_old = Variable(dtype, a_old)
                    if isinstance(a_new, (IndexedVariable,
                                  IndexedElement, str, Variable)):
                        v_new = Variable(dtype, a_new)
                    else:
                        v_new = a_new
                    expr = subs(expr, v_old, v_new)
                    inits[str(stmt.lhs)] = expr

    _ends = []
    for i in ends:
        if isinstance(i, IndexedElement):
            _ends.append(i.base)
        else:
            _ends.append(i)
    ends = [inits[str(i)] for i in _ends]

    # ...

    # ...

    if not len(ends) == len(starts):
        raise ValueError('wrong number of starts/ends')

    # ...

    return [Range(s, e, 1) for (s, e) in zip(starts, ends)]


# ...
from .numpyext import Linspace, Diag, Where
