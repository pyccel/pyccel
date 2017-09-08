# coding: utf-8

from __future__ import print_function, division

from numpy import ndarray

from sympy.core import Symbol, Tuple
from sympy.core.relational import Equality, Relational
from sympy.logic.boolalg import And, Boolean, Not, Or, true, false
from sympy.core.singleton import Singleton
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify
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

    Parameters
    ----------
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
    --------

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.printing.codeprinter import Assign
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
    """Represents augmented variable assignment for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : NativeOp
        Operator (+, -, /, *, %).

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    --------

    >>> from sympy import symbols
    >>> from pyccel.types.ast import AugAssign
    >>> x, y = symbols('x, y')
    >>> AugAssign(x, AddOp, y)
    x += y

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

    Parameters
    ----------
    test : expression
        test condition given as a sympy expression
    body : sympy expr
        list of statements representing the body of the While statement.
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

    Parameters
    ----------
    target : symbol
        symbol representing the iterator
    iter : iterable
        iterable object. for the moment only Range is used
    body : sympy expr
        list of statements representing the body of the For statement.
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

    Parameters
    ----------
    arg : str or sympy expression
        If a str ('bool', 'int', 'float', 'double', or 'void'), return the
        singleton for the corresponding dtype. If a sympy expression, return
        the datatype that best fits the expression. This is determined from the
        assumption system. For more control, use the `DataType` class directly.

    Returns
    -------
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

class Variable(Basic):
    """Represents a typed variable.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType, or a str (bool,
        int, float, double).
    name : Symbol, MatrixSymbol
        The sympy object the variable represents.
    rank : int
        used for arrays. [Default value: 0]
    allocatable: False
        used for arrays, if we need to allocate memory [Default value: False]

F    """
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
    """Argument provided as input only.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType, or a str (bool,
        int, float, double).
    name : Symbol, MatrixSymbol
        The sympy object the variable represents.

    """
    pass


class OutArgument(Argument):
    """OutputArgument are always initialized in the routine.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType, or a str (bool,
        int, float, double).
    name : Symbol, MatrixSymbol
        The sympy object the variable represents.

    """
    pass


class InOutArgument(Argument):
    """InOutArgument are never initialized in the routine.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType, or a str (bool,
        int, float, double).
    name : Symbol, MatrixSymbol
        The sympy object the variable represents.

    """
    pass


class FunctionDef(Basic):
    """Represents a function definition.

    Parameters
    ----------
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

class ceil(Basic):
    """Represents ceil expression in the code."""
    def __new__(cls,rhs):
        return Basic.__new__(cls,rhs)
    @property
    def rhs(self):
        return self._args[0]

class Import(Basic):
    """Represents inclusion of dependencies in the code.

    Parameters
    ----------
    fil : str
        The filepath of the module (i.e. header in C).
    funcs
        The name of the function (or an iterable of names) to be imported.

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

    Parameters
    ----------
    dtype : DataType
        The type for the declaration.
    variable(s)
        A single variable or an iterable of Variables. If iterable, all
        Variables must be of the same type.

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


class Return(Basic):
    """Represents a function return in the code.

    Parameters
    ----------
    expr : sympy expr
        The expression to return.

    """

    def __new__(cls, expr):
        expr = _sympify(expr)
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]


class LEN(Basic):
    """Represents a 'len' expression in the code."""
     def __new__(cls, rhs):
         return Basic.__new__(cls, rhs)
     @property
     def rhs(self):
        return self._args[0]
     @property
     def str(self):
        return 'size('+str(self._args[0])+',1)'

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

class Dot(Basic):
    """Represents a 'dot' expression in the code."""
     def __new__(cls, expr_l, expr_r):
         return Basic.__new__(cls, expr_l, expr_r)
     @property
     def expr_l(self):
         return self.args[0]
     @property
     def expr_r(self):
         return self.args[1]

class NumpyZeros(Basic):
    """Represents variable assignment using numpy.zeros for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers
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

    Parameters
    ----------
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

class NumpyArray(Basic):
    """Represents variable assignment using numpy.array for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
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


class NumpyLinspace(Basic):
    """Represents variable assignment using numpy.linspace for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers

    Examples
    --------

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.printing.codeprinter import Assign
    >>> x, y, z = symbols('x, y, z')
    >>> Assign(x, y)
    x := y

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

    Parameters
    ----------
    expr : sympy expr
        The expression to return.

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

    Parameters
    ----------
    text : str
       the comment line

    """

    def __new__(cls, text):
        return Basic.__new__(cls, text)

    @property
    def text(self):
        return self._args[0]

class AnnotatedComment(Basic):
    """Represents a Annotated Comment in the code.

    Parameters
    ----------
    accel : str
       accelerator id. One among {'omp', 'acc'}

    txt: str
        statement to print
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
    """Represents a Comment in the code.

    Parameters
    ----------
    text : str
       the comment line

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
    """Represents a Comment in the code.

    Parameters
    ----------
    text : str
       the comment line

    """

    def __new__(cls, base, *args, **kw_args):
#        print("args : ", args)
        return Indexed.__new__(cls, base, *args, **kw_args)

class Slice(Basic):
    """Represents a slice in the code.

    Parameters
    ----------
    start : Symbol or int
        starting index

    end : Symbol or int
        ending index

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

    Parameters
    ----------
    args :
        every argument is a tuple and
        is defined as (cond, expr) where expr is a valid ast element
        and cond is a boolean test.

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

    Parameters
    ----------
    start : Symbol or int
        starting index

    end : Symbol or int
        ending index

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

class Rational(Basic):
    """Represents a Rational numbers statement in the code.
    This is different from sympy.Rational, as it allows for symbolic numbers.

    Parameters
    ----------
    numerator : Symbol or int
        numerator of the Rational number

    denominator : Symbol or int
        denominator of the Rational number

    """
    def __new__(cls, numerator, denominator):
        return Basic.__new__(cls, numerator, denominator)

    @property
    def numerator(self):
        return self._args[0]

    @property
    def denominator(self):
        return self._args[1]

class Thread(Basic):
    """Represents a thread function for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    Examples
    --------

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


class ThreadID(Thread):
    """Represents a get thread id for code generation.
    """
    pass

class ThreadsNumber(Thread):
    """Represents a get threads number for code generation.
    """
    pass

class Stencil(Basic):
    """Represents variable assignment using a stencil for code generation.

    Parameters
    ----------
    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    shape : int or list of integers

    step : int or list of integers
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
