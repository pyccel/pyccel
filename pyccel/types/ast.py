# coding: utf-8

"""
Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.


AST Type Tree
-------------

*Basic*
     |--->Assign
     |--->AugAssign
     |--->NativeOp
     |           |--------------|
     |                          |--->AddOp
     |                          |--->SubOp
     |                          |--->MulOp
     |                          |--->DivOp
     |                          |--->ModOp
     |           *Singleton*----|
     |                    |
     |--->DataType        |
     |           |--------|--->NativeBool
     |                    |--->NativeInteger
     |                    |--->NativeFloat
     |                    |--->NativeDouble
     |                    |--->NativeVoid
     |
     |--->For
     |--->Variable
     |           |--->Argument
     |           |           |
     |           |           |--->InArgument
     |           |           |--->OutArgument
     |           |           |--->InOutArgument
     |           |--->Result
     |
     |--->FunctionDef
     |--->Import
     |--->Declare
     |--->Return
     |--->Print
     |--->Comment
"""

from __future__ import print_function, division


from sympy.core import Symbol, Tuple
from sympy.core.relational import Equality, Relational
from sympy.logic.boolalg import And, Boolean, Not, Or, true, false
from sympy.core.singleton import Singleton
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify
from sympy.core.compatibility import with_metaclass
from sympy.sets.fancysets import Range
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
from sympy.utilities.iterables import iterable



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
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ----------
    target : symbol
    iter : iterable
    body : sympy expr
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
    iter : iterable
    body : sympy expr
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


class NativeVoid(DataType):
    _name = 'Void'
    pass


Bool = NativeBool()
Int = NativeInteger()
Float = NativeFloat()
Double = NativeDouble()
Void = NativeVoid()


dtype_registry = {'bool': Bool,
                  'int': Int,
                  'float': Float,
                  'double': Double,
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
        if not shape==None:
            if  (not isinstance(shape,int) and not isinstance(shape,tuple) and not all(isinstance(n, int) for n in shape)):
                raise TypeError("shape must be an instance of int or tuple of int")



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
    """Represents a result directly returned from a routine.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType, or a str (bool,
        int, float, double).
    name : Symbol or MatrixSymbol, optional
        The sympy object the variable represents.

    """

    def __new__(cls, dtype, name=None):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")
        if not name:
            name = Symbol('')
        return Variable.__new__(cls, dtype, name)


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
        body = Tuple(*(_sympify(i) for i in body))
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

    Examples
    --------

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.printing.codeprinter import Assign
    >>> x, y, z = symbols('x, y, z')
    >>> Assign(x, y)
    x := y

    """

    # TODO improve in the spirit of assign
    def __new__(cls, lhs, shape):
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

class NumpyOnes(Basic):
    """

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

    def __new__(cls, lhs,rhs):
        lhs   = _sympify(lhs)


        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Idx)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
        if not isinstance(rhs,list):
            raise TypeError("cannot assign rhs of type %s." % type(rhs))

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

    do : bool
        True if an do section. [Default= False]

    end : bool
        True if an end section. [Default= False]

    parallel : bool
        True if a parallel section. [Default= False]

    section : str
        section to parallelize. One among {'for'}. [Default= '']

    visibility : str
        selected visibility. One among {'shared', 'private'}. [Default= 'shared']

    variables: list
        list of variables names. (and not Symbols!)

    """
    # TODO variables must be symbols

    # since some arguments may be None, we need to define their default values
    def __new__(cls, accel, do, end, parallel, section, visibility, variables):
        return Basic.__new__(cls, accel, do, end, parallel, section, visibility, variables)

    @property
    def accel(self):
        return self._args[0]

    @property
    def do(self):
        return self._args[1]

    @property
    def end(self):
        return self._args[2]

    @property
    def parallel(self):
        return self._args[3]

    @property
    def section(self):
        return self._args[4]

    @property
    def visibility(self):
        return self._args[5]

    @property
    def variables(self):
        return self._args[6]


class IndexedVariable(IndexedBase):
    """Represents a Comment in the code.

    Parameters
    ----------
    text : str
       the comment line

    """

    def __new__(cls, label, shape=None, **kw_args):
        return IndexedBase.__new__(cls, label, shape=None, **kw_args)

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
        return '{0} : {1}'.format(sstr(self.start), sstr(self.end))

class Piecewise(Basic):
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
