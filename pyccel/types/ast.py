# coding: utf-8

from __future__ import print_function, division

from numpy import ndarray

from sympy.core import Symbol, Tuple
from sympy.core.relational import Equality, Relational
from sympy.logic.boolalg import And, Boolean, Not, Or, true, false
from sympy.core.singleton import Singleton
from sympy.core.basic import Basic
# TODO rename _sympify to sympify. Before we were using _sympify from sympy.core
#      but then sympy will keep in memory all used variables. we don't need it,
#      since the in syntax.py we always check the namespace for any new variable.
from sympy import sympify as _sympify
from sympy.core.compatibility import with_metaclass
from sympy.core.compatibility import is_sequence
from sympy.sets.fancysets import Range
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
from sympy.utilities.iterables import iterable

# TODO: rename ceil to Ceil
# TODO: rename LEN to Len
# TODO: clean Thread objects
# TODO: update code examples
# TODO: add _sympystr whenever it's possible
__all__ = ["Assign", "NativeOp", "AddOp", "SubOp", "MulOp", "DivOp", \
           "ModOp", "AugAssign", "While", "For", "DataType", "NativeBool", \
           "NativeInteger", "NativeFloat", "NativeDouble", "NativeComplex", \
           "NativeVoid", "EqualityStmt", "NotequalStmt", "Variable", \
           "Argument", "Result", "InArgument", "OutArgument", \
           "InOutArgument", "FunctionDef", "ceil", "Import", "Declare", \
           "Return", "LEN", "Min", "Max", "Dot", \
           "NumpyZeros", "NumpyOnes", "NumpyArray", "NumpyLinspace", \
           "Print", "Comment", "AnnotatedComment", "IndexedVariable", \
           "IndexedElement", "Slice", "If", "MultiAssign", "Rational", \
           "Thread", "ThreadID", "ThreadsNumber", "Stencil"]

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

    def __new__(cls, lhs, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
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

    @property
    def expr(self):
        return self.rhs


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

    """

    def __new__(cls, lhs, op, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
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
        return Basic.__new__(cls, lhs, op, rhs)

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
        test = _sympify(test)

        if not iterable(body):
            raise TypeError("body must be an iterable")
        body = Tuple(*(_sympify(i) for i in body))
        return Basic.__new__(cls, test, body)

    @property
    def test(self):
        return self._args[0]


    @property
    def body(self):
        return self._args[1]

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

    def __new__(cls, target, iter, body):
        target = _sympify(target)
        if not iterable(iter):
            raise TypeError("iter must be an iterable")
        if type(iter) == tuple:
            # this is a hack, since Range does not accept non valued Integers.
#            r = Range(iter[0], 10000000, iter[2])
            r = Range(0, 10000000, 1)
            r._args = iter
            iter = r
        else:
            iter = _sympify(iter)

        if not iterable(body):
            raise TypeError("body must be an iterable")
        body = Tuple(*(_sympify(i) for i in body))
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
    pass


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
        arg = _sympify(arg)
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
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class NotequalStmt(Relational):
    """Represents a relational not equality expression in the code."""
    def __new__(cls,lhs,rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class GOrEq(Relational):
    def __new__(cls,lhs,rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class LOrEq(Relational):
    def __new__(cls,lhs,rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class Lthan(Relational):
    def __new__(cls,lhs,rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class Gter(Relational):
    def __new__(cls,lhs,rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        return Relational.__new__(cls,lhs,rhs)

class Variable(Basic):
    """Represents a typed variable.

    dtype : str, DataType
        The type of the variable. Can be either a DataType, or a str (bool,
        int, float, double).
    name : Symbol, MatrixSymbol
        The sympy object the variable represents.
    rank : int
        used for arrays. [Default value: 0]
    allocatable: False
        used for arrays, if we need to allocate memory [Default value: False]
    shape: int or list
        shape of the array. [Default value: None]

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Variable
    >>> x, n = symbols('x, n')
    >>> Variable('int', 'n')
    Variable(NativeInteger(), n, 0, False, None)
    >>> Variable('float', x, rank=2, shape=(n,2), allocatable=True)
    Variable(NativeFloat(), x, 2, True, (n, 2)
    """
    def __new__(cls, dtype, name, rank=0, allocatable=False,shape=None):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")
        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("Only Symbols and MatrixSymbols can be Variables.")
        if not isinstance(rank, int):
            raise TypeError("rank must be an instance of int.")
#        if not shape==None:
#            if  (not isinstance(shape,int) and not isinstance(shape,tuple) and not all(isinstance(n, int) for n in shape)):
#                raise TypeError("shape must be an instance of int or tuple of int")

        return Basic.__new__(cls, dtype, name, rank, allocatable,shape)

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

class Argument(Variable):
    """An abstract Argument data structure."""
    pass

class Result(Variable):
    """Represents a result directly returned from a routine."""
    pass

class InArgument(Argument):
    """Argument provided as input only."""
    pass

class OutArgument(Argument):
    """OutputArgument are always initialized in the routine."""
    pass

class InOutArgument(Argument):
    """InOutArgument are never initialized in the routine."""
    pass

class FunctionDef(Basic):
    """Represents a function definition.

    name : str
        The name of the function.
    arguments : iterable
        The arguments to the function, of type `Argument`.
    results : iterable
        The direct outputs of the function, of type `Result`.
    body : iterable
        The body of the function.
    local_vars : list of Symbols
        These are used internally by the routine.
    global_vars : list of Symbols
        Variables which will not be passed into the function.

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Assign, InArgument, Result, FunctionDef
    >>> n,x,y = symbols('n,x,y')
    >>> args        = [InArgument('float', x), InArgument('int', n)]
    >>> results     = [Result('float', y)]
    >>> body        = [Assign(y,x+n)]
    >>> local_vars  = []
    >>> global_vars = []
    >>> FunctionDef('f', args, results, body, local_vars, global_vars)
    FunctionDef(f, (InArgument(NativeFloat(), x, 0, False, None), InArgument(NativeInteger(), n, 0, False, None)), (Result(NativeFloat(), y, 0, False, None),), [y := n + x], [], [])
    """

    def __new__(cls, name, arguments, results, body, local_vars, global_vars):
        # name
        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, Symbol):
            raise TypeError("Function name must be Symbol or string")
        # arguments
        if not iterable(arguments):
            raise TypeError("arguments must be an iterable")
        if not all(isinstance(a, Argument) for a in arguments):
            raise TypeError("All arguments must be of type Argument")
        arguments = Tuple(*arguments)
        # body
        if not iterable(body):
            raise TypeError("body must be an iterable")
#        body = Tuple(*(i for i in body))
        # results
        if not iterable(results):
            raise TypeError("results must be an iterable")
        if not all(isinstance(i, Result) for i in results):
            raise TypeError("All results must be of type Result")
        results = Tuple(*results)

        # ...
        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set([])
        symbols = set([])
#        print("<>>>>> arguments : ", str(arguments))
        for arg in arguments:
            if isinstance(arg, OutArgument):
                symbols.update(arg.expr.free_symbols)
            elif isinstance(arg, InArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols)
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)

            try:
                symbols.update(r.expr.free_symbols)
            except:
                pass

            try:
                symbols.update(r.expr)
            except:
                pass

#        for stmt in statements:
#            if not isinstance(stmt, (Assign, Equality, For)):
#                raise ValueError("Unknown Routine statement: %s" % stmt)

        symbols = set([s.label if isinstance(s, Idx) else s for s in symbols])

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        notcovered = symbols.difference(
            input_symbols.union(local_vars).union(global_vars))

#        print (">>>>>>>>>>>>>>>>>>>> input_symbols :", str(input_symbols))
#        print (">>>>>>>>>>>>>>>>>>>> symbols       :", str(symbols))
#        print (">>>>>>>>>>>>>>>>>>>> not covered   :", str(notcovered))

#        if notcovered != set([]):
#            raise ValueError("Symbols needed for output are not in input or local " +
#                             ", ".join([str(x) for x in notcovered]))
        # ...


        return Basic.__new__(cls, name, \
                             arguments, results, \
                             body, \
                             local_vars, global_vars)

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

# TODO: rename and add example
class ceil(Basic):
    """
    Represents ceil expression in the code.

    rhs: symbol or number
        input for the ceil function
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
    """

    def __new__(cls, fil, funcs=None):
        fil = Symbol(fil)
        if not funcs:
            funcs = Tuple()
        elif iterable(funcs):
            funcs = Tuple(*[Symbol(f) for f in funcs])
        elif isinstance(funcs, str):
            funcs = Tuple(Symbol(funcs))
        else:
            raise TypeError("Unrecognized funcs type: ", funcs)
        return Basic.__new__(cls, fil, funcs)

    @property
    def fil(self):
        return self._args[0]

    @property
    def funcs(self):
        return self._args[1]

# TODO: Should Declare have an optional init value for each var?
class Declare(Basic):
    """Represents a variable declaration in the code.

    dtype : DataType
        The type for the declaration.
    variable(s)
        A single variable or an iterable of Variables. If iterable, all
        Variables must be of the same type.

    Examples

    >>> from sympy import Symbol
    >>> from pyccel.types.ast import Declare, Variable
    >>> n = Symbol('n')
    >>> var = Variable('int', 'n')
    >>> Declare('int', var)
    Declare(NativeInteger(), (Variable(NativeInteger(), n, 0, False, None),))
    """

    def __new__(cls, dtype, variables):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")
        if isinstance(variables, Variable):
            variables = [variables]
        for var in variables:
            if not isinstance(var, Variable):
                raise TypeError("var must be of type Variable")
            if var.dtype != dtype:
                raise ValueError("All variables must have the same dtype")
        variables = Tuple(*variables)
        return Basic.__new__(cls, dtype, variables)

    @property
    def dtype(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1]

# TODO: not used. do we keep it?
class Return(Basic):
    """Represents a function return in the code.

    expr : sympy expr
        The expression to return.

    """

    def __new__(cls, expr):
        expr = _sympify(expr)
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]
class Break(Basic):
    """Represents a function return in the code.

    Parameters
    ----------
    expr : sympy expr
        The expression to return.

    """

    def __new__(cls):
        return Basic.__new__(cls)

# TODO: rename and add example
class LEN(Basic):
    """
    Represents a 'len' expression in the code.

    rhs: symbol or number
        input for the len function
    """
    def __new__(cls, rhs):
        return Basic.__new__(cls, rhs)

    @property
    def rhs(self):
        return self._args[0]

    # TODO do we keep it? improve it
    @property
    def str(self):
        return 'size('+str(self._args[0])+',1)'

# TODO: improve by using args
# TODO: add example
class Min(Basic):
    """Represents a 'min' expression in the code."""
    def __new__(cls, expr_l, expr_r):
        return Basic.__new__(cls, expr_l, expr_r)

    @property
    def expr_l(self):
        return self.args[0]

    @property
    def expr_r(self):
        return self.args[1]

# TODO: improve by using args
# TODO: add example
class Max(Basic):
    """Represents a 'max' expression in the code."""
    def __new__(cls, expr_l, expr_r):
        return Basic.__new__(cls, expr_l, expr_r)

    @property
    def expr_l(self):
        return self.args[0]

    @property
    def expr_r(self):
        return self.args[1]

# TODO: add example
class Dot(Basic):
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

class SIGN(Basic):

    def __new__(cls,expr):
        return Basic.__new__(cls, expr)

    @property
    def rhs(self):
        return self.args[0]

class NumpyZeros(Basic):
    """Represents variable assignment using numpy.zeros for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers

    Examples

    >>> from sympy import symbols
    >>> from pyccel.types.ast import NumpyZeros
    >>> n,m,x = symbols('n,m,x')
    >>> NumpyZeros(x, (n,m))
    x := 0
    """
    # TODO improve in the spirit of assign
    def __new__(cls, lhs, shape):
        lhs   = _sympify(lhs)
        if isinstance(shape, list):
            # this is a correction. otherwise it is not working on LRZ
            if isinstance(shape[0], list):
                shape = Tuple(*(_sympify(i) for i in shape[0]))
            else:
                shape = Tuple(*(_sympify(i) for i in shape))
        elif isinstance(shape, int):
            shape = Tuple(_sympify(shape))
        elif isinstance(shape, Basic) and not isinstance(shape,LEN):
            shape = str(shape)
        elif isinstance(shape,LEN):
            shape=shape.str
        else:
            shape = shape

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        return Basic.__new__(cls, lhs, shape)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := 0'.format(sstr(self.lhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def shape(self):
        return self._args[1]

class NumpyOnes(Basic):
    """
    Represents variable assignment using numpy.ones for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers
    """
    # TODO improve in the spirit of assign
    def __new__(cls, lhs,shape):
        lhs   = _sympify(lhs)
        if isinstance(shape, list):
            shape = Tuple(*(_sympify(i) for i in shape))
        else:
            shape = shape

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        return Basic.__new__(cls, lhs, shape)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := 0'.format(sstr(self.lhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def shape(self):
        return self._args[1]

# TODO: add example
class NumpyArray(Basic):
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
        lhs   = _sympify(lhs)


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

# TODO: remove
class NumpyLinspace(Basic):
    """Represents variable assignment using numpy.linspace for code generation.

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    start: expression
        minimum of the grid
    end: expression
        maximum of the grid
    size: int, Expr
        number of elements of the grid
    """

    # TODO improve in the spirit of assign
    def __new__(cls, lhs, start, end, size):
        lhs   = _sympify(lhs)

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        return Basic.__new__(cls, lhs, start, end, size)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0} := 0'.format(sstr(self.lhs))

    @property
    def lhs(self):
        return self._args[0]

    @property
    def start(self):
        return self._args[1]

    @property
    def end(self):
        return self._args[2]

    @property
    def size(self):
        return self._args[3]

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
            expr = _sympify(expr)
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

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

    def __new__(cls, label, shape=None, **kw_args):
        return IndexedBase.__new__(cls, label, shape=None, **kw_args)

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
#            print ("indices : ", indices)
            return IndexedElement(self, *indices, **kw_args)
        else:
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return IndexedElement(self, indices, **kw_args)

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
        return Indexed.__new__(cls, base, *args, **kw_args)

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
    >>> x, y, z, t = symbols('x, y, z, t')
    >>> args = [x,y]
    >>> MultiAssign((z,t), 'f', args)
    z, t := f(x, y)
    """
    def __new__(cls, lhs, rhs, trailer):
        return Basic.__new__(cls, lhs, rhs, trailer)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def trailer(self):
        return self._args[2]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args    = ', '.join(sstr(i) for i in self.trailer)
        outputs = ', '.join(sstr(i) for i in self.lhs)
        return '{2} := {0}({1})'.format(self.rhs, args, outputs)

class Rational(Basic):
    """Represents a Rational numbers statement in the code.
    This is different from sympy.Rational, as it allows for symbolic numbers.

    numerator : Symbol or int
        numerator of the Rational number

    denominator : Symbol or int
        denominator of the Rational number

    >>> from sympy import symbols
    >>> from pyccel.types.ast import Rational
    >>> x, y, z = symbols('x, y, z')
    >>> a = x**2 + y*z
    >>> b = 2
    >>> Rational(a, b)
    (x**2 + y*z) / (2)
    """
    def __new__(cls, numerator, denominator):
        return Basic.__new__(cls, numerator, denominator)

    @property
    def numerator(self):
        return self._args[0]

    @property
    def denominator(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '({0}) / ({1})'.format(sstr(self.numerator), sstr(self.denominator))

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
        lhs   = _sympify(lhs)

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

# TODO: remove LEN from here
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
                    s_out = Tuple(*(_sympify(i) for i in s_in[0]))
                else:
                    s_out = Tuple(*(_sympify(i) for i in s_in))
            elif isinstance(s_in, int):
                s_out = Tuple(_sympify(s_in))
            elif isinstance(s_in, Basic) and not isinstance(s_in,LEN):
                s_out = str(s_in)
            elif isinstance(s_in,LEN):
                s_our = s_in.str
            else:
                s_out = s_in
            return s_out
        # ...

        # ...
        lhs   = _sympify(lhs)
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
