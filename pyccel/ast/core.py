#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
from itertools import chain

from sympy.logic.boolalg      import And as sp_And


from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import RECURSIVE_RESULTS_REQUIRED

from pyccel.utilities.stage   import PyccelStage
from pyccel.utilities.strings import create_incremented_string

from .basic     import Basic, PyccelAstNode, iterable, ScopedNode
from .builtins  import (PythonEnumerate, PythonLen, PythonMap, PythonTuple,
                        PythonRange, PythonZip, PythonBool, Lambda)
from .datatypes import (datatype, DataType, NativeSymbol,
                        NativeBool, NativeRange,
                        NativeTuple, str_dtype)
from .internals import Slice, PyccelSymbol, PyccelInternalFunction, get_final_precision

from .literals  import LiteralInteger, Nil, LiteralFalse
from .literals  import NilArgument, LiteralTrue

from .operators import PyccelAdd, PyccelMinus, PyccelMul, PyccelDiv, PyccelMod, Relational
from .operators import PyccelOperator, PyccelAssociativeParenthesis, PyccelIs

from .variable import DottedName, IndexedElement
from .variable import Variable

errors = Errors()
pyccel_stage = PyccelStage()

# TODO [YG, 12.03.2020]: Move non-Python constructs to other modules
# TODO [YG, 12.03.2020]: Rename classes to avoid name clashes in pyccel/ast
__all__ = (
    'AliasAssign',
    'Allocate',
    'AnnotatedComment',
    'AsName',
    'Assert',
    'Assign',
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
    'Deallocate',
    'Declare',
    'Decorator',
    'Del',
    'DottedFunctionCall',
    'Duplicate',
    'DoConcurrent',
    'EmptyNode',
    'ErrorExit',
    'Exit',
    'For',
    'ForIterator',
    'FuncAddressDeclare',
    'FunctionAddress',
    'FunctionCall',
    'FunctionCallArgument',
    'FunctionDef',
    'FunctionDefArgument',
    'FunctionDefResult',
    'If',
    'IfSection',
    'Import',
    'InlineFunctionDef',
    'InProgram',
    'Interface',
    'Iterable',
    'Module',
    'ModuleHeader',
    'Pass',
    'Program',
    'PyccelFunctionDef',
    'Raise',
    'Return',
    'SeparatorComment',
    'StarredArguments',
    'SymbolicAssign',
    'SymbolicPrint',
    'SympyFunction',
    'While',
    'With',
    'create_variable',
    'create_incremented_string',
    'get_iterable_ranges',
    'inline',
    'subs'
)

#==============================================================================

# TODO - add EmptyStmt => empty lines
#      - update code examples
#      - add examples
#      - Function case
#      - AnnotatedComment case
#      - add a new Idx that uses Variable instead of Symbol

#==============================================================================
def apply(func, args, kwargs):return func(*args, **kwargs)

#==============================================================================
def subs(expr, new_elements):
    """
    Substitutes old for new in an expression.

    Parameters
    ----------
    new_elements : list of tuples like [(x,2)(y,3)]
    """

    if len(list(new_elements)) == 0:
        return expr
    if isinstance(expr, (list, tuple)):
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
    elif isinstance(expr, PyccelAstNode):
        return expr.subs(new_elements)

    else:
        return expr

def inline(func, args):
    local_vars = func.local_vars
    body = func.body
    body = subs(body, zip(func.arguments, args))
    return Block(str(func.name), local_vars, body)

def create_variable(forbidden_names, prefix = None, counter = 1):
    """This function takes a prefix and a counter and uses them to construct
    a PyccelSymbol with a name of the form:
            prefix_counter
    Where counter is formatted to fill 4 characters
    The new name is checked against a list of forbidden expressions. If the
    constructed name is forbidden then the counter is incremented until a valid
    name is found

      Parameters
      ----------
      forbidden_exprs : Set
                        A set of all the values which are not valid solutions to this problem
      prefix          : str
                        The prefix used to begin the string
      counter         : int
                        The expected value of the next name

      Returns
      ----------
      name            : PyccelSymbol
                        A PyccelSymbol with the incremented string name
      counter         : int
                        The expected value of the next name

    """

    name, counter = create_incremented_string(forbidden_names, prefix, counter = counter)

    return PyccelSymbol(name, is_temp=True), counter


class AsName(Basic):

    """
    Represents a renaming of a variable, used with Import.

    Examples
    --------
    >>> from pyccel.ast.core import AsName, FunctionDef
    >>> from pyccel.ast.numpyext import NumpyFull
    >>> func = FunctionDef('old', (), (), ())
    >>> AsName(func, 'new')
    old as new
    >>> AsName(NumpyFull, 'fill_func')
    full as fill_func

    Parameters
    ==========
    obj    : Basic or BasicType
             The variable, function, or module being renamed
    target : str
             name of variable or function in this context
    """
    __slots__ = ('_obj', '_target')
    _attribute_nodes = ()

    def __init__(self, obj, target):
        if pyccel_stage != "syntactic":
            assert (isinstance(obj, Basic) and \
                    not isinstance(obj, PyccelSymbol)) or \
                   (isinstance(obj, type) and issubclass(obj, Basic))
        self._obj = obj
        self._target = target
        super().__init__()

    @property
    def name(self):
        """ The original name of the object
        """
        obj = self._obj
        if isinstance(obj, (str, PyccelSymbol, DottedName)):
            return obj
        else:
            return obj.name

    @property
    def target(self):
        """ The target name of the object
        """
        return self._target

    @property
    def object(self):
        """ The underlying object described by this AsName
        """
        return self._obj

    def __repr__(self):
        return '{0} as {1}'.format(str(self.name), str(self.target))

    def __str__(self):
        return '{0} as {1}'.format(str(self.name), str(self.target))

    def __eq__(self, string):
        if isinstance(string, str):
            return string == self.target
        elif isinstance(string, AsName):
            return string.target == self.target
        else:
            return self is string

    def __ne__(self, string):
        return not self == string

    def __hash__(self):
        return hash(self.target)


class Duplicate(PyccelAstNode):

    """ this is equivalent to the * operator for python tuples.

    Parameters
    ----------
    value : PyccelAstNode
           an expression which represents the initilized value of the list

    shape : the shape of the array
    """
    __slots__ = ('_val', '_length','_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_val', '_length')

    def __init__(self, val, length):
        self._dtype     = val.dtype
        self._precision = val.precision
        self._rank      = val.rank
        self._shape     = tuple(s if i!= 0 else PyccelMul(s, length, simplify=True) for i,s in enumerate(val.shape))
        self._order     = val.order

        self._val       = val
        self._length    = length
        super().__init__()

    @property
    def val(self):
        return self._val

    @property
    def length(self):
        return self._length

    def __str__(self):
        return '{} * {}'.format(str(self.val), str(self.length))

    def __repr__(self):
        return '{} * {}'.format(repr(self.val), repr(self.length))

class Concatenate(PyccelAstNode):

    """ this is equivalent to the + operator for python tuples

    Parameters
    ----------
    args : PyccelAstNodes
           The tuples
    """
    __slots__ = ('_args','_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_args',)

    def __init__(self, arg1, arg2):
        self._dtype     = arg1.dtype
        self._precision = arg1.precision
        self._rank      = arg1.rank
        shape_addition  = arg2.shape[0]
        self._shape     = tuple(s if i!= 0 else PyccelAdd(s, shape_addition) for i,s in enumerate(arg1.shape))
        self._order     = arg1.order

        self._args = (arg1, arg2)
        super().__init__()

    @property
    def args(self):
        return self._args


class Assign(Basic):

    """Represents variable assignment for code generation.

    Parameters
    ----------
    lhs : PyccelAstNode
        In the syntactic stage:
           Object representing the lhs of the expression. These should be
           singular objects, such as one would use in writing code. Notable types
           include PyccelSymbol, and IndexedElement. Types that
           subclass these types are also supported.
        In the semantic stage:
           Variable or IndexedElement

    rhs : PyccelAstNode
        In the syntactic stage:
          Object representing the rhs of the expression
        In the semantic stage :
          PyccelAstNode with the same shape as the lhs

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples
    --------
    >>> from pyccel.ast.internals import symbols
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import Assign
    >>> x, y, z = symbols('x, y, z')
    >>> Assign(x, y)
    x := y
    >>> Assign(x, 0)
    x := 0
    >>> A = Variable('int', 'A', rank = 2)
    >>> Assign(x, A)
    x := A
    >>> Assign(A[0,1], x)
    IndexedElement(A, 0, 1) := x
    """
    __slots__ = ('_lhs', '_rhs', '_status', '_like')
    _attribute_nodes = ('_lhs', '_rhs')

    def __init__(
        self,
        lhs,
        rhs,
        status=None,
        like=None,
        *,
        fst = None
        ):
        if isinstance(lhs, (tuple, list)):
            lhs = PythonTuple(*lhs)
        if isinstance(rhs, (tuple, list)):
            rhs = PythonTuple(*rhs)
        self._lhs = lhs
        self._rhs = rhs
        self._status = status
        self._like = like
        super().__init__()
        if fst is not None:
            self.set_fst(fst)

    def __str__(self):
        return '{0} := {1}'.format(str(self.lhs), str(self.rhs))

    def __repr__(self):
        return '({0} := {1})'.format(repr(self.lhs), repr(self.rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    # TODO : remove

    @property
    def expr(self):
        return self.rhs

    @property
    def status(self):
        return self._status

    @property
    def like(self):
        return self._like

    @property
    def is_alias(self):
        """Returns True if the assignment is an alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        cond = isinstance(rhs, Variable) and rhs.rank > 0
        cond = cond or isinstance(rhs, IndexedElement)
        cond = cond and isinstance(lhs, PyccelSymbol)
        cond = cond or isinstance(rhs, Variable) and rhs.is_alias
        return cond

    @property
    def is_symbolic_alias(self):
        """Returns True if the assignment is a symbolic alias."""

        # TODO to be improved when handling classes

        lhs = self.lhs
        rhs = self.rhs
        if isinstance(lhs, Variable):
            return isinstance(lhs.dtype, NativeSymbol)
        elif isinstance(lhs, PyccelSymbol):
            if isinstance(rhs, PythonRange):
                return True
            elif isinstance(rhs, Variable):
                return isinstance(rhs.dtype, NativeSymbol)
            elif isinstance(rhs, PyccelSymbol):
                return True

        return False

#------------------------------------------------------------------------------
class Allocate(Basic):
    """
    Represents memory allocation for code generation.

    Represents memory allocation (usually of an array) for code generation.
    This is relevant to low-level target languages, such as C or Fortran,
    where the programmer must take care of heap memory allocation.

    Parameters
    ----------
    variable : pyccel.ast.core.Variable
        The typed variable (usually an array) that needs memory allocation.

    shape : int or iterable or None
        Shape of the array after allocation (None for scalars).

    order : str {'C'|'F'}
        Ordering of multi-dimensional array after allocation
        ('C' = row-major, 'F' = column-major).

    status : str {'allocated'|'unallocated'|'unknown'}
        Variable allocation status at object creation.

    Notes
    -----
    An object of this class is immutable, although it contains a reference to a
    mutable Variable object.
    """
    __slots__ = ('_variable', '_shape', '_order', '_status')
    _attribute_nodes = ('_variable',)

    # ...
    def __init__(self, variable, *, shape, order, status):

        if not isinstance(variable, Variable):
            raise TypeError("Can only allocate a 'Variable' object, got {} instead".format(type(variable)))

        if variable.on_stack:
            # Variable may only be a pointer in the wrapper
            raise ValueError("Variable must be allocatable")

        if shape and not isinstance(shape, (int, tuple, list)):
            raise TypeError("Cannot understand 'shape' parameter of type '{}'".format(type(shape)))

        if variable.rank != len(shape):
            raise ValueError("Incompatible rank in variable allocation")

        # rank is None for lambda functions
        if variable.rank is not None and variable.rank > 1 and variable.order != order:
            raise ValueError("Incompatible order in variable allocation")

        if not isinstance(status, str):
            raise TypeError("Cannot understand 'status' parameter of type '{}'".format(type(status)))

        if status not in ('allocated', 'unallocated', 'unknown'):
            raise ValueError("Value of 'status' not allowed: '{}'".format(status))

        self._variable = variable
        self._shape    = shape
        self._order    = order
        self._status   = status
        super().__init__()
    # ...

    @property
    def variable(self):
        return self._variable

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def status(self):
        return self._status

    def __str__(self):
        return 'Allocate({}, shape={}, order={}, status={})'.format(
                str(self.variable), str(self.shape), str(self.order), str(self.status))

    def __eq__(self, other):
        if isinstance(other, Allocate):
            return (self.variable is other.variable) and \
                   (self.shape    == other.shape   ) and \
                   (self.order    == other.order   ) and \
                   (self.status   == other.status  )
        else:
            return False

    def __hash__(self):
        return hash((id(self.variable), self.shape, self.order, self.status))

#------------------------------------------------------------------------------
class Deallocate(Basic):
    """
    Represents memory deallocation (usually of an array) for code generation.
    This is relevant to low-level target languages, such as C or Fortran,
    where the programmer must take care of heap memory deallocation.

    Parameters
    ----------
    variable : pyccel.ast.core.Variable
        The typed variable (usually an array) that needs memory deallocation.

    Notes
    -----
    An object of this class is immutable, although it contains a reference to a
    mutable Variable object.

    """
    __slots__ = ('_variable',)
    _attribute_nodes = ('_variable',)

    # ...
    def __init__(self, variable):

        if not isinstance(variable, Variable):
            raise TypeError("Can only allocate a 'Variable' object, got {} instead".format(type(variable)))

        self._variable = variable
        super().__init__()

    # ...

    @property
    def variable(self):
        return self._variable

    def __eq__(self, other):
        if isinstance(other, Deallocate):
            return (self.variable is other.variable)
        else:
            return False

    def __hash__(self):
        return hash(id(self.variable))

#------------------------------------------------------------------------------
class CodeBlock(Basic):

    """Represents a list of stmt for code generation.
       we use it when a single statement in python
       produce multiple statement in the targeted language

       Parameters
       ==========
       body : iterable
    """
    __slots__ = ('_body','_unravelled')
    _attribute_nodes = ('_body',)


    def __init__(self, body, unravelled = False):
        ls = []
        for i in body:
            if isinstance(i, CodeBlock):
                ls += i.body
            elif i is not None and not isinstance(i, EmptyNode):
                ls.append(i)
        if not isinstance(unravelled, bool):
            raise TypeError("unravelled must be a boolean")
        self._body = tuple(ls)
        self._unravelled = unravelled
        super().__init__()

    @property
    def body(self):
        return self._body

    @property
    def unravelled(self):
        """ Indicates whether the vector syntax of python
        has been unravelled into for loops
        """
        return self._unravelled

    @property
    def lhs(self):
        return self.body[-1].lhs

    def insert2body(self, *obj, back=True):
        """ Insert object(s) to the body of the codeblock
        The object(s) are inserted at the back by default but
        can be inserted at the front by setting back to False
        """
        _ = [o.set_current_user_node(self) for o in obj]
        if back:
            self._body = tuple([*self.body, *obj])
        else:
            self._body = tuple([*obj, *self.body])

    def __repr__(self):
        return 'CodeBlock({})'.format(self.body)

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        kwargs = dict(body = self.body)
        return (apply, (self.__class__, (), kwargs))

    def set_fst(self, fst):
        super().set_fst(fst)
        for l in self.body:
            if not l.fst:
                l.set_fst(fst)

class AliasAssign(Basic):

    """Represents aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    Parameters
    ----------
    lhs : PyccelSymbol
        at this point we don't know yet all information about lhs, this is why a
        PyccelSymbol is the appropriate type.

    rhs : Variable, IndexedElement
        an assignable variable can be of any rank and any datatype, however its
        shape must be known (not None)

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import AliasAssign
    >>> from pyccel.ast.core import Variable
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x', rank=1, shape=[n])
    >>> y = PyccelSymbol('y')
    >>> AliasAssign(y, x)

    """
    __slots__ = ('_lhs','_rhs')
    _attribute_nodes = ('_lhs','_rhs')

    def __init__(self, lhs, rhs):
        if pyccel_stage == 'semantic':
            if not lhs.is_alias:
                raise TypeError('lhs must be a pointer')

            if isinstance(rhs, FunctionCall) and not rhs.funcdef.results[0].is_alias:
                raise TypeError("A pointer cannot point to the address of a temporary variable")

        self._lhs = lhs
        self._rhs = rhs
        super().__init__()

    def __str__(self):
        return '{0} := {1}'.format(str(self.lhs), str(self.rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs


class SymbolicAssign(Basic):

    """Represents symbolic aliasing for code generation. An alias is any statement of the
    form `lhs := rhs` where

    Parameters
    ----------
    lhs : PyccelSymbol

    rhs : Range

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import SymbolicAssign
    >>> from pyccel.ast.core import Range
    >>> r = Range(0, 3)
    >>> y = PyccelSymbol('y')
    >>> SymbolicAssign(y, r)

    """
    __slots__ = ('_lhs', '_rhs')
    _attribute_nodes = ('_lhs', '_rhs')

    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs
        super().__init__()

    def __str__(self):
        return '{0} := {1}'.format(str(self.lhs), str(self.rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs


class AugAssign(Assign):
    r"""
    Represents augmented variable assignment for code generation.

    Parameters
    ----------
    lhs : PyccelAstNode
        In the syntactic stage:
           Object representing the lhs of the expression. These should be
           singular objects, such as one would use in writing code. Notable types
           include PyccelSymbol, and IndexedElement. Types that
           subclass these types are also supported.
        In the semantic stage:
           Variable or IndexedElement

    op : str
        Operator (+, -, /, \*, %).

    rhs : PyccelAstNode
        In the syntactic stage:
          Object representing the rhs of the expression
        In the semantic stage :
          PyccelAstNode with the same shape as the lhs

    status: None, str
        if lhs is not allocatable, then status is None.
        otherwise, status is {'allocated', 'unallocated'}

    like: None, Variable
        contains the name of the variable from which the lhs will be cloned.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import AugAssign
    >>> s = Variable('int', 's')
    >>> t = Variable('int', 't')
    >>> AugAssign(s, '+', 2 * t + 1)
    s += 1 + 2*t
    """
    __slots__ = ('_op',)
    _accepted_operators = {
            '+' : PyccelAdd,
            '-' : PyccelMinus,
            '*' : PyccelMul,
            '/' : PyccelDiv,
            '%' : PyccelMod}

    def __init__(
        self,
        lhs,
        op,
        rhs,
        status=None,
        like=None,
        *,
        fst = None
        ):

        if op not in self._accepted_operators.keys():
            raise TypeError('Unrecognized Operator')

        self._op = op

        super().__init__(lhs, rhs, status, like, fst=fst)

    def __repr__(self):
        return '{0} {1}= {2}'.format(str(self.lhs), self.op, str(self.rhs))

    def __str__(self):
        return '{0} {1}= {2}'.format(str(self.lhs), self.op, str(self.rhs))

    @property
    def op(self):
        return self._op

    def to_basic_assign(self):
        """
        Convert the AugAssign to an Assign
        E.g. convert:
        a += b
        to:
        a = a + b
        """
        return Assign(self.lhs,
                self._accepted_operators[self._op](self.lhs, self.rhs),
                status = self.status,
                like   = self.like)


class While(ScopedNode):

    """Represents a 'while' statement in the code.

    Expressions are of the form:
        "while test:
            body..."

    Parameters
    ----------
    test : PyccelAstNode
        test condition given as an expression
    body : list of Pyccel objects
        list of statements representing the body of the While statement.
    scope : Scope
        The scope for the loop

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign, While
    >>> n = PyccelSymbol('n')
    >>> While((n>1), [Assign(n,n-1)])
    While(n > 1, (n := n - 1,))
    """
    __slots__ = ('_body','_test')
    _attribute_nodes = ('_body','_test')

    def __init__(self, test, body, scope = None):

        if pyccel_stage == 'semantic':
            if test.dtype is not NativeBool():
                test = PythonBool(test)

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')

        self._test = test
        self._body = body
        super().__init__(scope)

    @property
    def test(self):
        return self._test

    @property
    def body(self):
        return self._body

    @property
    def local_vars(self):
        """ List of variables defined in the loop """
        return tuple(self.scope.variables.values())


class With(ScopedNode):

    """Represents a 'with' statement in the code.

    Expressions are of the form:
        "with statement:
            body..."

    Parameters
    ----------
    test : PyccelAstNode
        with definition statement given as an expression
    body : list of Pyccel objects
        list of statements representing the body of the With statement.

    Examples
    --------

    """
    __slots__ = ('_test','_body')
    _attribute_nodes = ('_test','_body')

    # TODO check prelude and epilog

    def __init__(
        self,
        test,
        body,
        scope = None
        ):

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable')

        self._test = test
        self._body = body
        super().__init__(scope)

    @property
    def test(self):
        return self._test

    @property
    def body(self):
        return self._body

    @property
    def block(self):
        methods = self.test.cls_base.methods
        for i in methods:
            if str(i.name) == '__enter__':
                start = i
            elif str(i.name) == '__exit__':
                end   = i
        start = inline(start,[])
        end   = inline(end  ,[])

        # TODO check if enter is empty or not first

        body = start.body.body
        body += self.body.body
        body +=  end.body.body
        return Block('with', [], body, scope=self.scope)


# TODO add a name to a block?

class Block(ScopedNode):

    """Represents a block in the code. A block consists of the following inputs

    Parameters
    ----------
    variables: list
        list of the variables that appear in the block.

    declarations: list
        list of declarations of the variables that appear in the block.

    body: list
        a list of statements

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> Block([n, x], [Assign(x,2.*n + 1.), Assign(n, n + 1)])
    Block([n, x], [x := 1.0 + 2.0*n, n := 1 + n])
    """
    __slots__ = ('_name','_variables','_body')
    _attribute_nodes = ('_variables','_body')

    def __init__(
        self,
        name,
        variables,
        body,
        scope = None):
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
        self._name = name
        self._variables = variables
        self._body = body
        super().__init__(scope)

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        return self._variables

    @property
    def body(self):
        return self._body

    @property
    def declarations(self):
        return [Declare(i.dtype, i) for i in self.variables]



class Module(ScopedNode):
    """
    Represents a module in the code.

    The Pyccel node representing a Python module. A module consists of everything
    inside a given Python file.

    Parameters
    ----------
    name : str
        Name of the module.

    variables : list
        List of the variables that appear in the block.

    funcs : list
        A list of FunctionDef instances.

    init_func : FunctionDef, default: None
        The function which initialises the module (expressions in the
        python module which are executed on import).

    free_func : FunctionDef, default: None
        The function which frees any variables allocated in the module.

    program : Program/CodeBlock
        CodeBlock containing any expressions which are only executed
        when the module is executed directly.

    interfaces : list
        A list of Interface instances.

    classes : list
        A list of ClassDef instances.

    imports : list, tuple
        List of needed imports.

    scope : Scope
        The scope of the module.

    Examples
    --------
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import FunctionDefArgument, Assign, FunctionDefResult
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> from pyccel.ast.operators import PyccelAdd, PyccelMinus
    >>> from pyccel.ast.literals import LiteralInteger
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> z = Variable('float', 'z')
    >>> t = Variable('float', 't')
    >>> a = Variable('float', 'a')
    >>> b = Variable('float', 'b')
    >>> body = [Assign(z,PyccelAdd(x,a))]
    >>> args = [FunctionDefArgument(arg) for arg in [x,y,a,b]]
    >>> results = [FunctionDefResult(res) for res in [z,t]]
    >>> translate = FunctionDef('translate', args, results, body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> Point = ClassDef('Point', attributes, methods)
    >>> incr = FunctionDef('incr', [FunctionDefArgument(x)], [FunctionDefResult(y)], [Assign(y,PyccelAdd(x,LiteralInteger(1)))])
    >>> decr = FunctionDef('decr', [FunctionDefArgument(x)], [FunctionDefResult(y)], [Assign(y,PyccelMinus(x,LiteralInteger(1)))])
    >>> Module('my_module', [], [incr, decr], classes = [Point])
    Module(my_module, [], [FunctionDef(), FunctionDef()], [], [ClassDef(Point, (x, y), (FunctionDef(),), [public], (), [], [])], ())
    """
    __slots__ = ('_name','_variables','_funcs','_interfaces',
                 '_classes','_imports','_init_func','_free_func',
                 '_program','_variable_inits','_internal_dictionary')
    _attribute_nodes = ('_variables','_funcs','_interfaces',
                        '_classes','_imports','_init_func',
                        '_free_func','_program','_variable_inits')

    def __init__(
        self,
        name,
        variables,
        funcs,
        init_func = None,
        free_func = None,
        program = None,
        interfaces=(),
        classes=(),
        imports=(),
        scope = None
        ):
        if not isinstance(name, (str, AsName)):
            raise TypeError('name must be a string or an AsName')

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
                raise TypeError('Only a Interface instance is allowed.')

        NoneType = type(None)
        if not isinstance(init_func, (NoneType, FunctionDef)):
            raise TypeError('init_func must be a FunctionDef')

        if not isinstance(free_func, (NoneType, FunctionDef)):
            raise TypeError('free_func must be a FunctionDef')

        if not isinstance(program, (NoneType, Program, CodeBlock)):
            raise TypeError('program must be a Program (or a CodeBlock at the syntactic stage)')

        if not iterable(imports):
            raise TypeError('imports must be an iterable')
        imports = list(imports)
        for i in classes:
            imports += i.imports
        imports = set(imports)  # for unicity
        imports = tuple(imports)

        self._name = name
        self._variables = variables
        self._variable_inits = [None]*len(variables)
        self._funcs = funcs
        self._init_func = init_func
        self._free_func = free_func
        self._program   = program
        self._interfaces = interfaces
        self._classes = classes
        self._imports = imports

        self._internal_dictionary = {v.name:v for v in variables}
        self._internal_dictionary.update({f.name:f for f in funcs})
        self._internal_dictionary.update({i.name:i for i in interfaces})
        self._internal_dictionary.update({c.name:c for c in classes})
        import_mods = {i.source: [t.object for t in i.target if isinstance(t.object, Module)] for i in imports}
        self._internal_dictionary.update({v:t[0] for v,t in import_mods.items() if t})

        if init_func:
            init_if = init_func.body.body[0]
            # The init function should always contain an If block unless it is part of a wrapper
            if isinstance(init_if, If):
                init_cond = init_if.blocks[0].condition
                init_var = init_cond.args[0]
                self._variables.append(init_var)
                self._variable_inits.append(LiteralFalse())

        super().__init__(scope)

    @property
    def name(self):
        """ Name of the module
        """
        return self._name

    @property
    def variables(self):
        """ Module global variables
        """
        return self._variables

    @property
    def init_func(self):
        """ The function which initialises the module (expressions in the
        python module which are executed on import)
        """
        return self._init_func

    @property
    def free_func(self):
        """ The function which frees any variables allocated in the module
        """
        return self._free_func

    @property
    def program(self):
        """ CodeBlock or Program containing any expressions which are only executed
        when the module is executed directly
        """
        return self._program

    @program.setter
    def program(self, prog):
        assert self._program is None
        self._program = prog
        self._program.set_current_user_node(self)

    @property
    def funcs(self):
        """ Any functions defined in the module
        """
        return self._funcs

    @property
    def interfaces(self):
        """ Any interfaces defined in the module
        """
        return self._interfaces

    @property
    def classes(self):
        """ Any classes defined in the module
        """
        return self._classes

    @property
    def imports(self):
        """ Any imports in the module
        """
        return self._imports

    @property
    def declarations(self):
        """ Returns the declarations of the variables
        """
        return [Declare(i.dtype, i, value=v, module_variable=True) \
                for i,v in zip(self.variables, self._variable_inits)]

    @property
    def body(self):
        """ Returns the functions, interfaces and classes defined
        in the module
        """
        return self.interfaces + self.funcs + self.classes

    def set_name(self, new_name):
        """ Function for changing the name of a module
        """
        self._name = new_name

    def __getitem__(self, arg):
        assert isinstance(arg, str)
        args = arg.split('.')
        result = self._internal_dictionary[args[0]]
        for key in args[1:]:
            result = result[key]
        return result

    def __contains__(self, arg):
        assert isinstance(arg, (str, PyccelSymbol, DottedName))
        args = str(arg).split('.')
        current_pos = self._internal_dictionary
        key = args[0]
        result = key in self._internal_dictionary
        i = 1
        while i<len(args) and result:
            current_pos = current_pos[key]
            key = args[i]
            result = key in current_pos
            i += 1
        return result

    def keys(self):
        """ Returns the names of all objects accessible directly in this module
        """
        return self._internal_dictionary.keys()

class ModuleHeader(Basic):
    """
    Represents the header file for a module.

    This class is simply a wrapper around a module. It is helpful to differentiate
    between headers and sources when printing.

    Parameters
    ----------
    module : Module
        The module described by the header.

    See Also
    --------
    Module : The module itself.

    Examples
    --------
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import FunctionDefArgument, Assign, FunctionDefResult
    >>> from pyccel.ast.core import ClassDef, FunctionDef, Module
    >>> from pyccel.ast.operators import PyccelAdd, PyccelMinus
    >>> from pyccel.ast.literals import LiteralInteger
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> z = Variable('float', 'z')
    >>> t = Variable('float', 't')
    >>> a = Variable('float', 'a')
    >>> b = Variable('float', 'b')
    >>> body = [Assign(z,PyccelAdd(x,a))]
    >>> args = [FunctionDefArgument(arg) for arg in [x,y,a,b]]
    >>> results = [FunctionDefResult(res) for res in [z,t]]
    >>> translate = FunctionDef('translate', args, results, body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> Point = ClassDef('Point', attributes, methods)
    >>> incr = FunctionDef('incr', [FunctionDefArgument(x)], [FunctionDefResult(y)], [Assign(y,PyccelAdd(x,LiteralInteger(1)))])
    >>> decr = FunctionDef('decr', [FunctionDefArgument(x)], [FunctionDefResult(y)], [Assign(y,PyccelMinus(x,LiteralInteger(1)))])
    >>> Module('my_module', [], [incr, decr], classes = [Point])
    >>> ModuleHeader(mod)
    Module(my_module, [], [FunctionDef(), FunctionDef()], [], [ClassDef(Point, (x, y), (FunctionDef(),), [public], (), [], [])], ())
    """
    __slots__ = ('_module',)
    _attribute_nodes = ('_module',)

    def __init__(self, module):
        if not isinstance(module, Module):
            raise TypeError('module must be a Module')

        self._module = module
        super().__init__()

    @property
    def module(self):
        return self._module

class Program(ScopedNode):

    """Represents a Program in the code. A block consists of the following inputs

    Parameters
    ----------
    variables: list
        list of the variables that appear in the block.

    body: list
        a list of statements

    imports: list, tuple
        list of needed imports

    """
    __slots__ = ('_name', '_variables', '_body', '_imports')
    _attribute_nodes = ('_variables', '_body', '_imports')

    def __init__(
        self,
        name,
        variables,
        body,
        imports=(),
        scope=None
        ):

        if not isinstance(name, str):
            raise TypeError('name must be a string')

        if not iterable(variables):
            raise TypeError('variables must be an iterable')

        for i in variables:
            if not isinstance(i, Variable):
                raise TypeError('Only a Variable instance is allowed.')

        if not iterable(body):
            raise TypeError('body must be an iterable')
        body = CodeBlock(body)

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        imports = set(imports)  # for unicity
        imports = tuple(imports)

        self._name = name
        self._variables = variables
        self._body = body
        self._imports = imports
        super().__init__(scope)

    @property
    def name(self):
        """ Name of the executable
        """
        return self._name

    @property
    def variables(self):
        """ Variables contained within the program
        """
        return self._variables

    @property
    def body(self):
        """ Statements in the program
        """
        return self._body

    @property
    def imports(self):
        """ Imports imported in the program
        """
        return self._imports

    def remove_import(self, name):
        """ Remove an import with the given source name from the list
        of imports
        """
        _ = [i.invalidate_node() for i in self.imports if i.source == name]
        self._imports = tuple(i for i in self.imports if i.source != name)


#==============================================================================
class Iterable(Basic):
    """
    Wrapper around iterable types helping to convert between those
    types and a range (necessary in low level languages, e.g. C and Fortran)

    Paramaters
    ----------
    iterable : acceptable_iterator_type
               The iterator being wrapped
               The type must be in acceptable_iterator_types or the class must
               implement the following functions:
               - n_indices
               - to_range
    """
    acceptable_iterator_types = (Variable, PythonMap, PythonZip, PythonEnumerate, PythonRange, IndexedElement)
    __slots__ = ('_iterable','_indices','_num_indices_required')
    _attribute_nodes = ('_iterable','_indices')

    def __init__(self, iterable):
        self._iterable = iterable
        self._indices  = None

        if isinstance(iterable, PythonRange):
            self._num_indices_required = 0
        elif isinstance(iterable, PythonEnumerate):
            self._num_indices_required = int(iterable.start != 0)
        elif isinstance(iterable, self.acceptable_iterator_types):
            self._num_indices_required = 1
        elif hasattr(iterable, 'n_indices') and hasattr(iterable, 'to_range'):
            self._num_indices_required = iterable.n_indices
        else:
            raise TypeError("Unknown iterator type {}".format(type(iterable)))

        super().__init__()

    @property
    def num_loop_counters_required(self):
        """ Number of iterators which must be generated in order to
        convert this iterable to a range
        """
        return self._num_indices_required

    def set_loop_counter(self, *indices):
        """ Set the iterator(s) for the generated range
        These are iterators generated by pyccel so are not
        needed for python code
        """
        assert self._indices is None
        for i in indices:
            i.set_current_user_node(self)
        self._indices = indices

    def unset_loop_counter(self, invalidate = True):
        """ Remove the iterator(s) set for the generated range
        """
        for i in self._indices:
            i.remove_user_node(self, invalidate)
        self._indices = None

    def get_assigns(self, target):
        """ Returns a list containing any assigns necessary to initialise
        the loop iterators/targets when using a range iterable

        Parameters
        ----------
        target : Variable or list of Variables
                 The index(es) over which the loop iterates

        Results
        -------
        assigns : list of Assign
                  The assignments necessary to define target
        """
        iterable = self._iterable
        if isinstance(iterable, PythonRange):
            return []
        range_element = self.get_target_from_range()
        if self.num_loop_counters_required == 0:
            target = target[1:]
            range_element = range_element[1:]
        if isinstance(target, (tuple, list)):
            return [AliasAssign(t, r) if t.is_alias else Assign(t, r) for t, r in zip(target, range_element)]
        else:
            return [AliasAssign(target, range_element) if target.is_alias else Assign(target, range_element)]

    def get_target_from_range(self):
        """ Returns an element of the range indexed with the iterators
        previously provided via the set_loop_counters method
        (useful for get_assigns and to determine the dtype etc of the
        loop iterator)
        """
        idx = self._indices[0] if len(self._indices)==1 else self._indices
        range_base = self._iterable[idx]
        if isinstance(self._iterable, PythonMap):
            return FunctionCall(range_base[0], [range_base[1]])
        else:
            return range_base

    def get_range(self):
        """ Returns the range required for this iterable
        """
        if isinstance(self._iterable, PythonRange):
            return self._iterable
        elif hasattr(self._iterable, 'to_range'):
            return self._iterable.to_range()
        else:
            length = getattr(self._iterable, '__len__',
                    getattr(self._iterable, 'length', PythonLen(self._iterable)))
            if callable(length):
                length = length()
            return PythonRange(length)

    @property
    def iterable(self):
        """ Returns the iterable being wrapped
        """
        return self._iterable

    @property
    def loop_counters(self):
        """ Returns the iterator(s) of the generated range
        """
        return self._indices

#==============================================================================

class For(ScopedNode):

    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ----------
    target : symbol / Variable
        symbol representing the iterator
    iter : iterable
        iterable object. for the moment only Range is used
    body : list of pyccel objects
        list of statements representing the body of the For statement.
    scope : Scope
        The scope for the loop

    Examples
    --------
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import Assign, For
    >>> from pyccel.ast.internals import symbols
    >>> i,b,e,s,x = symbols('i,b,e,s,x')
    >>> A = Variable('int', 'A', rank = 2)
    >>> For(i, (b,e,s), [Assign(x, i), Assign(A[0, 1], x)])
    For(i, (b, e, s), (x := i, IndexedElement(A, 0, 1) := x))
    """
    __slots__ = ('_target','_iterable','_body','_end_annotation')
    _attribute_nodes = ('_target','_iterable','_body')

    def __init__(
        self,
        target,
        iter_obj,
        body,
        scope = None
        ):
        if pyccel_stage != "syntactic":
            if not isinstance(iter_obj, Iterable):
                iter_obj = Iterable(iter_obj)
                if iter_obj.num_loop_counters_required!=0:
                    raise TypeError('iter_obj must be an iterable')

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a Codeblock')

        self._target = target
        self._iterable = iter_obj
        self._body = body
        self._end_annotation = None
        super().__init__(scope)

    @property
    def end_annotation(self):
        return self._end_annotation

    @end_annotation.setter
    def end_annotation(self, expr):
        self._end_annotation = expr

    @property
    def target(self):
        return self._target

    @property
    def iterable(self):
        return self._iterable

    @property
    def body(self):
        return self._body

    @property
    def local_vars(self):
        """ List of variables defined in the loop """
        return tuple(self.scope.variables.values())

    def insert2body(self, stmt):
        stmt.set_current_user_node(self)
        self.body.insert2body(stmt)



class DoConcurrent(For):
    __slots__ = ()
    pass


class ForIterator(For):

    """Class that describes iterable classes defined by the user."""
    __slots__ = ()

    def __init__(
        self,
        target,
        iterable,
        body,
        ):

        if isinstance(iterable, Variable):
            iterable = PythonRange(PythonLen(iterable))
        super().__init__(target, iterable, body)

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

            cls_base = it.cls_base
            if not cls_base:
                raise TypeError('cls_base undefined')

            methods = cls_base.methods_as_dict
            it_method = methods['__iter__']

            it_vars = []
            for stmt in it_method.body:
                if isinstance(stmt, Assign):
                    it_vars.append(stmt.lhs)

            n = len(set(str(var.name) for var in it_vars))
            return n
        else:

            return 1

    @property
    def ranges(self):
        return get_iterable_ranges(self.iterable)

class FunctionCallArgument(Basic):
    """
    An argument passed in a function call

    Parameters
    ---------
    value   : PyccelAstNode
              The expression passed as an argument
    keyword : str
              If the argument is passed by keyword then this
              is that keyword
    """
    __slots__ = ('_value', '_keyword')
    _attribute_nodes = ('_value',)
    def __init__(self, value, keyword = None):
        self._value = value
        self._keyword = keyword
        super().__init__()

    @property
    def value(self):
        """ The value passed as argument
        """
        return self._value

    @property
    def keyword(self):
        """ The keyword used to pass the argument
        """
        return self._keyword

    @property
    def has_keyword(self):
        """ Indicates whether the argument was passed by keyword
        """
        return self._keyword is not None

    def __repr__(self):
        if self.has_keyword:
            return 'FunctionCallArgument({} = {})'.format(self.keyword, repr(self.value))
        else:
            return 'FunctionCallArgument({})'.format(repr(self.value))

    def __str__(self):
        if self.has_keyword:
            return '{} = {}'.format(self.keyword, str(self.value))
        else:
            return '{}'.format(str(self.value))

class FunctionDefArgument(PyccelAstNode):
    """
    Node describing the argument of a function.

    An object describing the argument of a function described
    by a FunctionDef. This object stores all the information
    which describes an argument but is superfluous for a Variable.

    Parameters
    ----------
    name : PyccelSymbol, Variable, FunctionAddress
        The name of the argument.

    value : PyccelAstNode, default: None
        The default value of the argument.

    kwonly : bool
        Indicates if the argument must be passed by keyword.

    annotation : str
        The type annotation describing the argument.

    See Also
    --------
    FunctionDef : The class where these objects will be stored.

    Examples
    --------
    >>> from pyccel.ast.core import FunctionDefArgument
    >>> n = FunctionDefArgument('n')
    >>> n
    n
    """
    __slots__ = ('_name','_var','_kwonly','_annotation','_value','_inout')
    _attribute_nodes = ('_value','_var')

    def __init__(self, name, *, value = None, kwonly=False, annotation=None):
        if isinstance(name, (Variable, FunctionAddress)):
            self._var  = name
            self._name = name.name
        elif isinstance(name, PyccelSymbol):
            self._var  = name
            self._name = name
        else:
            raise TypeError("Name must be a PyccelSymbol, Variable or FunctionAddress")
        self._value      = value
        self._kwonly     = kwonly
        self._annotation = annotation

        if isinstance(name, Variable):
            name.declare_as_argument()

        if pyccel_stage != "syntactic":
            self._inout = self.var.rank>0 and not self.var.is_const if isinstance(self.var, Variable) else False

        super().__init__()

    @property
    def name(self):
        """ The name of the argument
        """
        return self._name

    @property
    def var(self):
        """ The variable representing the argument
        (available after the semantic treatment)
        """
        return self._var

    @property
    def is_kwonly(self):
        """ Indicates if the argument must be passed
        by keyword
        """
        return self._kwonly

    @property
    def annotation(self):
        """ The argument annotation providing dtype information
        """
        return self._annotation

    @property
    def value(self):
        """ The default value of the argument
        """
        return self._value

    @property
    def default_call_arg(self):
        """ The FunctionCallArgument which is passed to FunctionCall
        if no value is provided for this argument
        """
        return FunctionCallArgument(self.value, keyword=self.name) \
                if self.has_default else None

    @property
    def has_default(self):
        """ Indicates whether the argument has a default value
        (if not then it must be provided)
        """
        return self._value is not None

    @property
    def inout(self):
        """
        Indicates whether the argument may be modified by the function.

        True if the argument may be modified in the function. False if
        the argument remains constant in the function.
        """
        return self._inout

    def make_const(self):
        """
        Indicate that the argument does not change in the function.

        Indicate that the argument does not change in the function by
        modifying the inout flag.
        """
        self._inout = False

    def __str__(self):
        if self.has_default:
            argument = str(self.name)
            value = str(self.value)
            return '{0}={1}'.format(argument, value)
        else:
            return str(self.name)

    def __repr__(self):
        if self.has_default:
            argument = str(self.name)
            value = str(self.value)
            return 'FunctionDefArgument({0}={1})'.format(argument, value)
        else:
            return 'FunctionDefArgument({})'.format(repr(self.name))

class FunctionDefResult(PyccelAstNode):
    """
    Node describing the result of a function.

    An object describing the result of a function described
    by a FunctionDef. This object stores all the information
    which describes an result but is superfluous for a Variable.

    Parameters
    ----------
    var : Variable
        The variable which represents the returned value.

    annotation : str, default: None
        The type annotation describing the argument.

    See Also
    --------
    FunctionDef : The class where these objects will be stored.

    Examples
    --------
    >>> from pyccel.ast.core import FunctionDefResult
    >>> n = FunctionDefResult('n')
    >>> n
    n
    """
    __slots__ = ('_var','_is_argument','_annotation')
    _attribute_nodes = ('_var',)

    def __init__(self, var, *, annotation=None):
        self._var        = var
        self._annotation = annotation

        if pyccel_stage == 'syntactic':
            if not isinstance(var, PyccelSymbol):
                raise TypeError("Var must be a PyccelSymbol")
        elif not isinstance(var, Variable):
            raise TypeError("Var must be a Variable")
        else:
            self._is_argument = var.is_argument

        super().__init__()

    @property
    def var(self):
        """
        The variable representing the result.

        The variable which represents the result. This variable is only
        available after the semantic stage.
        """
        return self._var

    @property
    def annotation(self):
        """
        The result annotation providing dtype information.

        The annotation which provides all information about the data
        types, precision, etc, necessary to fully define the result.
        """
        return self._annotation

    @property
    def is_argument(self):
        """
        Indicates if the result was declared as an argument.

        Indicates if the result of the function was initially declared
        as an argument of the same function. If this is the case then
        the result may be printed simply as an inout argument.
        """
        return self._is_argument

    def __repr__(self):
        return 'FunctionDefResult({})'.format(repr(self.var))

    def __str__(self):
        return str(self.var)

class FunctionCall(PyccelAstNode):
    """
    Represents a function call in the code.

    A node which holds all information necessary to represent a function
    call in the code.

    Parameters
    ----------
    func : FunctionDef
        The function being called.

    args : list of FunctionCallArgument
        The arguments passed to the function.

    current_function : FunctionDef, default: None
        The function where the call takes place.
    """
    __slots__ = ('_arguments','_funcdef','_interface','_func_name','_interface_name',
                 '_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_arguments','_funcdef','_interface')

    def __init__(self, func, args, current_function=None):

        for a in args:
            assert not isinstance(a, FunctionDefArgument)
        # Ensure all arguments are of type FunctionCallArgument
        args = [a if isinstance(a, FunctionCallArgument) else FunctionCallArgument(a) for a in args]

        if pyccel_stage == "syntactic":
            self._interface = None
            self._funcdef   = func
            self._arguments = tuple(args)
            self._func_name = func
            super().__init__()
            return

        # ...
        if not isinstance(func, (FunctionDef, Interface)):
            raise TypeError('> expecting a FunctionDef or an Interface')

        if isinstance(func, Interface):
            self._interface = func
            self._interface_name = func.name
            func = func.point(args)
        else:
            self._interface = None

        name = func.name
        # ...
        if isinstance(current_function, DottedName):
            current_function = current_function.name[-1]
        if current_function == name:
            func.set_recursive()

        if not isinstance(args, (tuple, list)):
            raise TypeError('args must be a list or tuple')

        # add the missing argument in the case of optional arguments
        f_args = func.arguments
        if not len(args) == len(f_args):
            # Collect dict of keywords and values (initialised as default)
            f_args_dict = {a.name: (a.name, a.value) if a.has_default \
                    else None for a in f_args}
            keyword_args = []
            for i,a in enumerate(args):
                if a.keyword is None:
                    # Replace default positional arguments with provided arguments
                    f_args_dict[f_args[i].name] = a
                else:
                    keyword_args = args[i:]
                    break

            for a in keyword_args:
                # Replace default arguments with provided keyword arguments
                f_args_dict[a.keyword] = a

            args = [FunctionCallArgument(keyword=a[0], value=a[1]) if isinstance(a, tuple) else a for a in f_args_dict.values()]

        # Handle function as argument
        arg_vals = [None if a is None else a.value for a in args]
        args = [FunctionCallArgument(FunctionAddress(av.name, av.arguments, av.results, []), keyword=a.keyword)
                if isinstance(av, FunctionDef) else a for a, av in zip(args, arg_vals)]

        if current_function == func.name:
            if len(func.results)>0 and not isinstance(func.results[0], PyccelAstNode):
                errors.report(RECURSIVE_RESULTS_REQUIRED, symbol=func, severity="fatal")

        self._funcdef       = func
        self._arguments     = args
        self._dtype         = func.results[0].var.dtype     if len(func.results) == 1 else NativeTuple()
        self._rank          = func.results[0].var.rank      if len(func.results) == 1 else None
        self._shape         = func.results[0].var.shape     if len(func.results) == 1 else None
        self._precision     = func.results[0].var.precision if len(func.results) == 1 else None
        self._order         = func.results[0].var.order     if len(func.results) == 1 else None
        self._func_name     = func.name
        super().__init__()

    @property
    def args(self):
        """ List of FunctionCallArguments provided to the function call
        (contains default values after semantic stage)
        """
        return self._arguments

    @property
    def funcdef(self):
        """ The function called by this function call
        """
        return self._funcdef

    @property
    def interface(self):
        """ The interface called by this function call
        """
        return self._interface

    @property
    def func_name(self):
        """ The name of the function called by this function call
        """
        return self._func_name

    @property
    def interface_name(self):
        """ The name of the interface called by this function call
        """
        return self._interface_name

    def __repr__(self):
        return '{}({})'.format(self.func_name, ', '.join(str(a) for a in self.args))

    @classmethod
    def _ignore(cls, c):
        """ Indicates if a node should be ignored when recursing
        """
        return c is None or isinstance(c, (FunctionDef, *cls._ignored_types))

class DottedFunctionCall(FunctionCall):
    """
    Represents a function call in the code where
    the function is defined in another object
    (e.g. module/class)

    a.f()

    Parameters
    ==========
    func             : FunctionDef
                       The definition of the function being called
    args             : tuple
                       The arguments being passed to the function
    prefix           : PyccelAstNode
                       The object in which the function is defined
                       E.g. for a.f()
                       prefix will contain a
    current_function : str
                        The function from which this call occurs
                        (This is required in order to recognise
                        recursive functions)
    """
    __slots__ = ('_prefix',)
    _attribute_nodes = (*FunctionCall._attribute_nodes, '_prefix')

    def __init__(self, func, args, prefix, current_function=None):
        self._prefix = prefix
        super().__init__(func, args, current_function)
        self._func_name = DottedName(prefix, self._func_name)
        if self._interface:
            self._interface_name = DottedName(prefix, self._interface_name)

    @property
    def prefix(self):
        """ The object in which the function is defined
        """
        return self._prefix

class ConstructorCall(DottedFunctionCall):

    """
    Represents a Constructor call in the code.

    A node which holds all information necessary to represent a Constructor
    call in the code.

    Parameters
    ----------
    func : FunctionDef, str
        An instance of FunctionDef or function name.

    arguments : list, tuple, None
        A list of arguments.

    cls_variable : CustomDataType, optional
        The variable on the left-hand side of an assignment,
        where the right-hand side is a constructor call.
        Used to store data inside the class, set during object creation.
    """
    __slots__ = ('_cls_variable',)
    _attribute_nodes = ()

    # TODO improve

    def __init__(
        self,
        func,
        arguments,
        cls_variable=None,
        ):
        if not isinstance(func, (FunctionDef, Interface, str)):
            raise TypeError('Expecting func to be a FunctionDef or str')

        self._cls_variable = cls_variable
        super().__init__(func, arguments, self._cls_variable)

    @property
    def cls_variable(self):
        """
        Get the class variable associated with the constructor.

        The `cls_variable` property allows accessing the class
        variable associated with the constructor.

        Returns
        -------
        CustomDataType or None
            The class variable associated with the constructor, or None if not provided.
        """
        return self._cls_variable


class Return(Basic):

    """Represents a function return in the code.

    Parameters
    ----------
    expr : PyccelAstNode
        The expression to return.

    stmts :represent assign stmts in the case of expression return
    """
    __slots__ = ('_expr', '_stmt')
    _attribute_nodes = ('_expr', '_stmt')

    def __init__(self, expr, stmt=None):

        if stmt and not isinstance(stmt, CodeBlock):
            raise TypeError('stmt should only be of type CodeBlock')

        self._expr = expr
        self._stmt = stmt

        super().__init__()

    @property
    def expr(self):
        return self._expr

    @property
    def stmt(self):
        return self._stmt

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.expr, self.stmt)
        return args

    def __repr__(self):
        if self.stmt:
            code = repr(self.stmt)+';'
        else:
            code = ''
        return code+"Return({})".format(','.join([repr(e) for e in self.expr]))

class FunctionDef(ScopedNode):

    """
    Represents a function definition.

    Node containing all the information necessary to describe a function.
    This information should provide enough information to print a functionally
    equivalent function in any target language.

    Parameters
    ----------
    name : str
        The name of the function.

    arguments : iterable of FunctionDefArgument
        The arguments to the function.

    results : iterable
        The direct outputs of the function.

    body : iterable
        The body of the function.

    global_vars : list of Symbols
        Variables which will not be passed into the function.

    cls_name : str
        Class name if the function is a method of cls_name.

    is_static : bool
        True for static functions. Needed for iso_c_binding interface.

    imports : list, tuple
        A list of needed imports.

    decorators : dict
        A dictionary whose keys are the names of decorators and whose values
        contain their implementation.

    headers : list,tuple
        A list of headers describing the function.

    is_recursive : bool
        True for a function which calls itself.

    is_pure : bool
        True for a function without side effect.

    is_elemental : bool
        True for a function that is elemental.

    is_private : bool
        True for a function that is private.

    is_header : bool
        True for a function which has no body available.

    is_external : bool
        True for a function which cannot be explicitly imported or renamed.

    functions : list, tuple
        A list of functions defined within this function.

    interfaces : list, tuple
        A list of interfaces defined within this function.

    doc_string : str
        The doc string of the function.

    scope : parser.scope.Scope
        The scope containing all objects scoped to the inside of this function.

    See Also
    --------
    FunctionDefArgument : The type used to store the arguments.

    Examples
    --------
    >>> from pyccel.ast.variable import Variable
    >>> from pyccel.ast.core import FunctionDefArgument, FunctionDefResult
    >>> from pyccel.ast.core import Assign, FunctionDef
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.literals import LiteralInteger
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> args        = [FunctionDefArgument(x)]
    >>> results     = [FunctionDefResult(y)]
    >>> body        = [Assign(y,PyccelAdd(x,LiteralInteger(1)))]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x,), (y,), [y := x + 1], [], [], None, False, function)

    One can also use parametrized argument, using FunctionDefArgument

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Assign
    >>> from pyccel.ast.core import FunctionDef
    >>> from pyccel.ast.core import FunctionDefArgument
    >>> n = FunctionDefArgument('n', value=4)
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> args        = [x, n]
    >>> results     = [y]
    >>> body        = [Assign(y,x+n)]
    >>> FunctionDef('incr', args, results, body)
    FunctionDef(incr, (x, n=4), (y,), [y := 1 + x], [], [], None, False, function, [])
    """
    __slots__ = ('_name','_arguments','_results','_body',
                 '_global_vars','_cls_name','_is_static','_imports',
                 '_decorators','_headers','_is_recursive','_is_pure',
                 '_is_elemental','_is_private','_is_header',
                 '_functions','_interfaces','_doc_string', '_is_external')
    _attribute_nodes = ('_arguments','_results','_body',
                 '_global_vars','_imports','_functions','_interfaces')

    def __init__(
        self,
        name,
        arguments,
        results,
        body,
        global_vars=(),
        cls_name=None,
        is_static=False,
        imports=(),
        decorators={},
        headers=(),
        is_recursive=False,
        is_pure=False,
        is_elemental=False,
        is_private=False,
        is_header=False,
        is_external=False,
        functions=(),
        interfaces=(),
        doc_string=None,
        scope=None):

        if isinstance(name, str):
            name = PyccelSymbol(name)
        elif isinstance(name, (tuple, list)):
            name_ = []
            for i in name:
                if isinstance(i, str):
                    name_.append(PyccelSymbol(i))
                else:
                    raise TypeError('Function name must be PyccelSymbol or string'
                                    )
            name = tuple(name_)
        else:
            raise TypeError('Function name must be PyccelSymbol or string')

        # arguments

        if not iterable(arguments):
            raise TypeError('arguments must be an iterable')
        if not all(isinstance(a, FunctionDefArgument) for a in arguments):
            raise TypeError('arguments must be all be FunctionDefArguments')

        arg_vars = [a.var for a in arguments]

        # body

        if iterable(body):
            body = CodeBlock(body)
        elif not isinstance(body,CodeBlock):
            raise TypeError('body must be an iterable or a CodeBlock')

#        body = tuple(i for i in body)
        # results

        if not iterable(results):
            raise TypeError('results must be an iterable')
        if not all(isinstance(r, FunctionDefResult) for r in results):
            raise TypeError('results must be all be FunctionDefResults')

        # if method

        if cls_name:

            if not isinstance(cls_name, str):
                raise TypeError('cls_name must be a string')

            # if not cls_variable:
             #   raise TypeError('Expecting a instance of {0}'.format(cls_name))

        if not isinstance(is_static, bool):
            raise TypeError('Expecting a boolean for is_static attribute')

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
            raise TypeError('Expecting a boolean for header')

        if not isinstance(is_external, bool):
            raise TypeError('Expecting a boolean for external')
        else:
            is_external = is_external and is_header and ( len(results) == 1 )

        if functions:
            for i in functions:
                if not isinstance(i, FunctionDef):
                    raise TypeError('Expecting a FunctionDef')

        self._name            = name
        self._arguments       = arguments
        self._results         = results
        self._body            = body
        self._global_vars     = global_vars
        self._cls_name        = cls_name
        self._is_static       = is_static
        self._imports         = imports
        self._decorators      = decorators
        self._headers         = headers
        self._is_recursive    = is_recursive
        self._is_pure         = is_pure
        self._is_elemental    = is_elemental
        self._is_private      = is_private
        self._is_header       = is_header
        self._is_external     = is_external
        self._functions       = functions
        self._interfaces      = interfaces
        self._doc_string      = doc_string
        super().__init__(scope)

    @property
    def name(self):
        """ Name of the function """
        return self._name

    @property
    def arguments(self):
        """ List of variables which are the function arguments """
        return self._arguments

    @property
    def results(self):
        """ List of variables which are the function results """
        return self._results

    @property
    def body(self):
        """ CodeBlock containing all the statements in the function """
        return self._body

    @property
    def local_vars(self):
        """
        List of variables defined in the function.

        A list of all variables which are local to the function. This
        includes arguments, results, and variables defined inside the
        function.
        """
        local_vars = self.scope.variables.values()
        argument_vars = [a.var for a in self.arguments]
        result_vars = [r.var for r in self.results]
        return tuple(l for l in local_vars if l not in result_vars and l not in argument_vars)

    @property
    def global_vars(self):
        """ List of global variables used in the function """
        return self._global_vars

    @property
    def cls_name(self):
        """ String containing the name of the class to which the method belongs.
        If the function is not a class procedure then this returns None """
        return self._cls_name

    @cls_name.setter
    def cls_name(self, cls_name):
        self._cls_name = cls_name

    @property
    def imports(self):
        """ List of imports in the function """
        return self._imports

    @property
    def decorators(self):
        """ List of decorators applied to the function """
        return self._decorators

    @property
    def headers(self):
        """ List of headers applied to the function """
        return self._headers

    @property
    def templates(self):
        """ List of templates used to determine the types """
        return self._templates

    @property
    def is_recursive(self):
        """ Returns True if the function is recursive (i.e. calls itself)
        and False otherwise """
        return self._is_recursive

    @property
    def is_pure(self):
        """ Returns True if the function is marked as pure and False otherwise
        Pure functions must not have any side effects.
        In other words this means that the result must be the same no matter
        how many times the function is called
        e.g:
        >>> a = f()
        >>> a = f()

        gives the same result as
        >>> a = f()

        This is notably not true for I/O functions
        """
        return self._is_pure

    @property
    def is_elemental(self):
        """ returns True if the function is marked as elemental and
        False otherwise
        An elemental function is a function with a single scalar operator
        and a scalar return value which can also be called on an array.
        When it is called on an array it returns the result of the function
        called elementwise on the array """
        return self._is_elemental

    @property
    def is_private(self):
        """ True if the function should not be exposed to
        other modules. This includes the wrapper module and
        means that the function cannot be used in an import
        or exposed to python """
        return self._is_private

    @property
    def is_header(self):
        """ True if the implementation of the function body
        is not provided False otherwise """
        return self._is_header

    @property
    def is_external(self):
        """ True if the function is exposed through a header file and coming
        from a f77 module """
        return self._is_external

    @property
    def is_inline(self):
        """ True if the function should be printed inline """
        return False

    @property
    def functions(self):
        """ List of functions within this function """
        return self._functions

    @property
    def interfaces(self):
        """ List of interfaces within this function """
        return self._interfaces

    @property
    def doc_string(self):
        """ The docstring of the function """
        return self._doc_string

    def set_recursive(self):
        """ Mark the function as a recursive function """
        self._is_recursive = True

    def clone(self, newname, **new_kwargs):
        """
        Create an almost identical FunctionDef with name `newname`.

        Create an almost identical FunctionDef with name `newname`.
        Additional parameters can be passed to alter the resulting
        FunctionDef.

        Parameters
        ----------
        newname : str
            New name for the FunctionDef.

        **new_kwargs : dict
            Any new keyword arguments to be passed to the new FunctionDef.

        Returns
        -------
        FunctionDef
            The clone of the function definition.
        """
        args, kwargs = self.__getnewargs__()
        kwargs.update(new_kwargs)
        cls = type(self)
        new_func = cls(*args, **kwargs)
        new_func.rename(newname)
        return new_func


    def rename(self, newname):
        """
        Rename the FunctionDef name
        newname.

        Parameters
        ----------
        newname: str
            new name for the FunctionDef
        """

        self._name = newname

    def __getnewargs__(self):
        """
          This method returns the positional and keyword arguments
            used to create an instance of this class.
        """
        args = (
        self._name,
        self._arguments,
        self._results,
        self._body)

        kwargs = {
        'global_vars':self._global_vars,
        'cls_name':self._cls_name,
        'is_static':self._is_static,
        'imports':self._imports,
        'decorators':self._decorators,
        'headers':self._headers,
        'is_recursive':self._is_recursive,
        'is_pure':self._is_pure,
        'is_elemental':self._is_elemental,
        'is_private':self._is_private,
        'is_header':self._is_header,
        'functions':self._functions,
        'is_external':self._is_external,
        'interfaces':self._interfaces,
        'doc_string':self._doc_string,
        'scope':self._scope}
        return args, kwargs

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        args, kwargs = self.__getnewargs__()
        out = (apply, (self.__class__, args, kwargs))
        return out


    def __str__(self):
        result = 'None' if len(self.results) == 0 else \
                    ', '.join(str(r) for r in self.results)
        args = ', '.join(str(a) for a in self.arguments)
        return '{name}({args}) -> {result}'.format(
                name   = self.name,
                args   = args,
                result = result)

    @property
    def is_unused(self):
        return False

class InlineFunctionDef(FunctionDef):
    """
    Represents a function definition for an inline function.

    Parameters
    ----------
    See FunctionDef

    namespace_imports : Scope
                        The objects in the scope which are available due to imports
    """
    __slots__ = ('_namespace_imports','_orig_args','_new_args','_new_local_vars', '_if_block_replacements',
            '_global_funcs')

    def __init__(self, *args, namespace_imports = None, global_funcs = None, **kwargs):
        self._namespace_imports = namespace_imports
        self._global_funcs = global_funcs
        super().__init__(*args, **kwargs)
        self._orig_args = tuple(a.var for a in self.arguments)
        self._new_args  = None
        self._new_local_vars = None
        self._if_block_replacements = None

    @property
    def is_inline(self):
        """ True if the function should be printed inline """
        return True

    @property
    def namespace_imports(self):
        """ The objects in the scope which are available due to imports
        """
        return self._namespace_imports

    def swap_in_args(self, args, new_local_vars):
        """ Modify the body of the function by replacing the arguments
        and local variables with the provided arguments and local variables
        """
        assert self._new_args is None
        assert self._new_local_vars is None
        # Collect the function arguments and the expressions they will be replaced with
        self._new_local_vars = tuple(new_local_vars)

        # We cannot replace with singletons as this cannot be reversed
        self._new_args  = tuple(NilArgument() if a is Nil() else \
                        PyccelAssociativeParenthesis(a) if isinstance(a, PyccelOperator) \
                        else a for a in args)

        # Replace the arguments in the code
        self.body.substitute(self._orig_args+self.local_vars, self._new_args+self._new_local_vars, invalidate=False)

    def swap_out_args(self):
        """ Modify the body of the function by reinstating the original arguments and local variables
        """
        self.body.substitute(self._new_args+self._new_local_vars, self._orig_args+self.local_vars, invalidate=False)
        self._new_args = None
        self._new_local_vars = None

    def remove_presence_checks(self):
        """ Modify the body by replacing all expressions checking for the presence of an optional
        variable. Either the If is removed or the check is replaced with its literal result
        """
        assert self._if_block_replacements is None
        # Look for if blocks and replace present(x) statements
        if_blocks = self.body.get_attribute_nodes(If, excluded_nodes=(FunctionDef,))
        if_block_replacements = [[], []]
        for i in if_blocks:
            blocks = []
            for c,e in i.blocks:
                if isinstance(c, PyccelIs):
                    if c.eval() is True:
                        blocks.append((LiteralTrue(), e))
                        break
                    elif c.eval() is False:
                        continue
                else:
                    presence_checks = c.get_attribute_nodes(PyccelIs, excluded_nodes = (FunctionDef,))
                    for pi in presence_checks:
                        if pi.eval() is True:
                            if_block_replacements[0].append(pi)
                            if_block_replacements[1].append(LiteralTrue())
                        elif pi.eval() is False:
                            if_block_replacements[0].append(pi)
                            if_block_replacements[1].append(LiteralFalse())
                blocks.append((c, e))
            if len(blocks) == 0:
                if_block_replacements[0].append(i)
                if_block_replacements[1].append(EmptyNode())
            elif len(blocks) == 1 and isinstance(blocks[0][0], LiteralTrue):
                if_block_replacements[0].append(i)
                if_block_replacements[1].append(blocks[0][1])
            elif len(blocks) != len(i.blocks):
                if_block_replacements[0].append(i)
                if_block_replacements[1].append(If(*blocks))
        self._if_block_replacements = if_block_replacements
        self.body.substitute(if_block_replacements[0], if_block_replacements[1], invalidate=False)

    def reinstate_presence_checks(self):
        """ Modify the body by reinstating all expressions checking for the presence of an optional
        variable
        """
        self.body.substitute(self._if_block_replacements[1], self._if_block_replacements[0])
        self._if_block_replacements = None

    @property
    def global_funcs(self):
        """ List of global functions used in the function """
        return self._global_funcs

class PyccelFunctionDef(FunctionDef):
    """
    Class used for storing `PyccelInternalFunction` objects in a FunctionDef.

    Class inheriting from `FunctionDef` which can store a pointer
    to a class type defined by pyccel for treating internal functions.
    This is useful for importing builtin functions and for defining
    classes which have `PyccelInternalFunction`s as attributes or methods.

    Parameters
    ----------
    name : str
        The name of the function.

    func_class : type inheriting from PyccelInternalFunction / PyccelAstNode
        The class which should be instantiated upon a FunctionCall
        to this FunctionDef object.

    decorators : dictionary
        A dictionary whose keys are the names of decorators and whose values
        contain their implementation.

    argument_description : dict, optional
        A dictionary containing all arguments and their default values. This
        is useful in order to reuse types with similar functionalities but
        different default values.
    """
    __slots__ = ('_argument_description',)
    def __init__(self, name, func_class, *, decorators = {}, argument_description = {}):
        assert isinstance(func_class, type) and \
                issubclass(func_class, (PyccelInternalFunction, PyccelAstNode))
        assert isinstance(argument_description, dict)
        arguments = ()
        results = ()
        body = ()
        super().__init__(name, arguments, results, body, decorators=decorators)
        self._cls_name = func_class
        self._argument_description = argument_description

    @property
    def argument_description(self):
        """
        Get a description of the arguments.

        Return a dictionary whose keys are the arguments with default values
        and whose values are the default values for the function described by
        the `PyccelFunctionDef`
        """
        return self._argument_description

class Interface(Basic):

    """Represents an Interface.

    Parameters
    ----------
    name : str
        The name of the interface.

    functions : iterable
        The functions of the interface.

    is_argument: bool
        True if the interface is used for a function argument.

    Examples
    --------
    >>> from pyccel.ast.core import Interface, FunctionDef
    >>> f = FunctionDef('F', [], [], [])
    >>> Interface('I', [f])
    """
    __slots__ = ('_name','_functions','_is_argument')
    _attribute_nodes = ('_functions',)

    def __init__(
        self,
        name,
        functions,
        is_argument = False,
        ):

        if not isinstance(name, str):
            raise TypeError('Expecting an str')
        if not isinstance(functions, list):
            raise TypeError('Expecting a list')
        self._name = name
        self._functions = functions
        self._is_argument = is_argument
        super().__init__()

    @property
    def name(self):
        """Name of the interface."""
        return self._name

    @property
    def functions(self):
        """"Functions of the interface."""
        return self._functions

    @property
    def is_argument(self):
        """True if the interface is used for a function argument."""
        return self._is_argument

    @property
    def doc_string(self):
        return self._functions[0].doc_string

    def point(self, args, use_final_precision = False):
        """Returns the actual function that will be called, depending on the passed arguments."""
        fs_args = [[j for j in i.arguments] for i in
                    self._functions]

        if use_final_precision:
            type_match = lambda dtype1, dtype2, call_arg, func_arg: \
                    (dtype1 in dtype2 or dtype2 in dtype1) \
                    and (call_arg.rank == func_arg.rank) \
                    and get_final_precision(call_arg) == \
                        get_final_precision(func_arg)
        else:
            type_match = lambda dtype1, dtype2, call_arg, func_arg: \
                    (dtype1 in dtype2 or dtype2 in dtype1) \
                    and (call_arg.rank == func_arg.rank) \
                    and call_arg.precision == func_arg.precision


        j = -1
        for i in fs_args:
            j += 1
            found = True
            for (x, y) in enumerate(args):
                func_arg = i[x].var
                call_arg = y.value
                dtype1 = str_dtype(call_arg.dtype)
                dtype2 = str_dtype(func_arg.dtype)
                found = found and type_match(dtype1, dtype2, call_arg, func_arg)
            if found:
                break

        if found:
            return  self._functions[j]
        else:
            errors.report('Arguments types provided to {} are incompatible'.format(self.name),
                        severity='fatal')

class FunctionAddress(FunctionDef):

    """Represents a function address.

    Parameters
    ----------
    name : str
        The name of the function address.

    arguments : iterable
        The arguments to the function address.

    results : iterable
        The direct outputs of the function address.

    is_argument: bool
        if object is the argument of a function [Default value: False]

    is_kwonly: bool
        if object is an argument which can only be specified using its keyword

    is_optional: bool
        if object is an optional argument of a function [Default value: False]

    memory_handling: str
        must be 'heap', 'stack' or 'alias' [Default value: 'stack']

    Examples
    --------
    >>> from pyccel.ast.core import Variable, FunctionAddress, FuncAddressDeclare, FunctionDef
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')

    a function definition can have a FunctionAddress as an argument

    >>> FunctionDef('g', [FunctionAddress('f', [x], [y], [])], [], [])

    we can also Declare a FunctionAddress

    >>> FuncAddressDeclare(FunctionAddress('f', [x], [y], []))
    """
    __slots__ = ('_is_optional','_is_kwonly','_is_argument', '_memory_handling')

    def __init__(
        self,
        name,
        arguments,
        results,
        body,
        is_optional=False,
        is_kwonly=False,
        is_argument=False,
        memory_handling='stack',
        **kwargs
        ):
        super().__init__(name, arguments, results, body, scope=1,**kwargs)
        if not isinstance(is_argument, bool):
            raise TypeError('Expecting a boolean for is_argument')

        if memory_handling not in ('heap', 'alias', 'stack'):
            raise TypeError('Expecting \'heap\', \'stack\', \'alias\' or None for memory_handling')

        if not isinstance(is_kwonly, bool):
            raise TypeError('Expecting a boolean for kwonly')

        elif not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')

        self._is_optional   = is_optional
        self._is_kwonly     = is_kwonly
        self._is_argument   = is_argument
        self._memory_handling = memory_handling

    @property
    def name(self):
        return self._name

    @property
    def memory_handling(self):
        """ Returns the memory handling of the instance of FunctionAddress
        """
        return self._memory_handling

    @property
    def is_alias(self):
        """ Indicates if the instance of FunctionAddress is an alias
        """
        return self.memory_handling == 'alias'

    @property
    def is_argument(self):
        return self._is_argument

    @property
    def is_kwonly(self):
        return self._is_kwonly

    @property
    def is_optional(self):
        return self._is_optional

class SympyFunction(FunctionDef):

    """Represents a function definition."""
    __slots__ = ()



class ClassDef(ScopedNode):
    """
    Represents a class definition.

    Class representing a class definition in the code. It holds all objects
    which may be defined in a class including methods, interfaces, attributes,
    etc. It also handles inheritance.

    Parameters
    ----------
    name : str
        The name of the class.

    attributes : iterable
        The attributes to the class.

    methods : iterable
        Class methods.

    options : list, tuple
        A list of options ('public', 'private', 'abstract').

    imports : list, tuple
        A list of required imports.

    superclasses : iterable
        The definition of all classes from which this class inherits.

    interfaces : iterable
        The interface methods.

    scope : Scope
        The scope for the class contents.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.core import ClassDef, FunctionDef
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> z = Variable('float', 'z')
    >>> t = Variable('float', 't')
    >>> a = Variable('float', 'a')
    >>> b = Variable('float', 'b')
    >>> body = [Assign(y,x+a)]
    >>> translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
    >>> attributes   = [x,y]
    >>> methods     = [translate]
    >>> ClassDef('Point', attributes, methods)
    ClassDef(Point, (x, y), (FunctionDef(translate, (x, y, a, b), (z, t), [y := a + x], [], [], None, False, function),), [public])
    """
    __slots__ = ('_name','_attributes','_methods','_options',
                 '_imports','_superclasses','_interfaces', '_is_deallocated')
    _attribute_nodes = ('_attributes', '_methods', '_imports', '_interfaces')

    def __init__(
        self,
        name,
        attributes=(),
        methods=(),
        options=('public',),
        imports=(),
        superclasses=(),
        interfaces=(),
        scope = None
        ):

        # name

        if isinstance(name, str):
            name = PyccelSymbol(name)
        else:
            raise TypeError('Function name must be PyccelSymbol or string')

        # attributes

        if not iterable(attributes):
            raise TypeError('attributes must be an iterable')
        attributes = tuple(attributes)

        # methods

        if not iterable(methods):
            raise TypeError('methods must be an iterable')

        # options

        if not iterable(options):
            raise TypeError('options must be an iterable')

        # imports

        if not iterable(imports):
            raise TypeError('imports must be an iterable')

        if not iterable(superclasses):
            raise TypeError('superclasses must be iterable')
        if pyccel_stage != 'syntactic':
            for s in superclasses:
                if not isinstance(s, ClassDef):
                    raise TypeError('superclass item must be a ClassDef')

        if not iterable(interfaces):
            raise TypeError('interfaces must be iterable')

        imports = list(imports)
        for i in methods:
            imports += list(i.imports)

        imports = set(imports)  # for unicity
        imports = tuple(imports)

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
         #   for a in attributes:
         #       if isinstance(a, Variable):
         #           if a.allocatable:
         #              args.append(a)

         #   args = [Variable(a.dtype, DottedName(str(this), str(a.name))) for a in args]
         #   body = [Del(a) for a in args]

         #   free = FunctionDef('__del__', [this], [], \
         #                      body, local_vars=[], global_vars=[], \
         #                      cls_name='__UNDEFINED__', imports=[])

         #  methods = list(methods) + [free]
         # TODO move this somewhere else

        methods = tuple(methods)

        # ...
        self._name = name
        self._attributes = attributes
        self._methods = methods
        self._options = options
        self._imports = imports
        self._superclasses  = superclasses
        self._interfaces = interfaces
        self._is_deallocated = False

        super().__init__(scope = scope)

    @property
    def name(self):
        return self._name

    @property
    def attributes(self):
        """
        The attributes of a class.

        Returns a tuple containing the attributes of a ClassDef.
        Each element within the tuple is of type Variable.
        """
        return self._attributes

    @property
    def is_deallocated(self):
        """
        Indicates whether the class has been deallocated.

        Returns True if the class instance has been deallocated, False otherwise.
        """

        return self._is_deallocated

    @is_deallocated.setter
    def is_deallocated(self, value):
        if not isinstance(value, bool):
            raise TypeError("is_deallocated must be a boolean (Bool)")
        self._is_deallocated = value

    @property
    def methods(self):
        return self._methods

    @property
    def options(self):
        return self._options

    @property
    def imports(self):
        return self._imports

    @property
    def superclasses(self):
        """
        Get the superclasses.

        Get the class definitions for the classes from which this class
        inherits.
        """
        return self._superclasses

    @property
    def interfaces(self):
        return self._interfaces

    @property
    def methods_as_dict(self):
        """Returns a dictionary that contains all methods, where the key is the
        method's name."""

        d_methods = {}
        for i in self.methods:
            d_methods[i.name] = i
        return d_methods

    @property
    def attributes_as_dict(self):
        """Returns a dictionary that contains all attributes, where the key is the
        attribute's name."""

        d_attributes = {}
        for i in self.attributes:
            d_attributes[i.name] = i
        return d_attributes

    def add_new_attribute(self, attr):
        """
        Add a new attribute to the current class.

        Add a new attribute to the current ClassDef.

        Parameters
        ----------
        attr : Variable
            The Variable that will be added.
        """

        if not isinstance(attr, Variable):
            raise TypeError("Attributes must be Variables")
        attr.set_current_user_node(self)
        self._attributes += (attr,)

    def add_new_method(self, method):
        """
        Add a new method to the current class.

        Add a new method to the current ClassDef.

        Parameters
        ----------
        method : FunctionDef
            The Method that will be added.
        """

        if not isinstance(method, FunctionDef):
            raise TypeError("Method must be FunctionDef")
        method.set_current_user_node(self)
        self._methods += (method,)

    def add_new_interface(self, interface):
        """
        Add a new interface to the current class.

        Add a new interface to the current ClassDef.

        Parameters
        ----------
        interface : FunctionDef
            The interface that will be added.
        """

        if not isinstance(interface, Interface):
            raise TypeError("Argument 'interface' must be of type Interface")
        interface.set_current_user_node(self)
        self._interfaces += (interface,)

    def get_attribute(self, O, attr):
        """Returns the attribute attr of the class O of instance self."""

        if not isinstance(attr, str):
            raise TypeError('Expecting attribute to be a string')

        if isinstance(O, Variable):
            cls_name = O.name
        else:
            cls_name = str(O)

        attributes = {}
        for i in self.attributes:
            attributes[i.name] = i

        if not attr in attributes:
            raise ValueError('{0} is not an attribute of {1}'.format(attr,
                             str(self)))

        var = attributes[attr]
        name = DottedName(cls_name, var.name)
        return Variable(
            var.dtype,
            name,
            rank=var.rank,
            memory_handling=var.memory_handling,
            shape=var.shape,
            cls_base=var.cls_base,
            )

    def get_method(self, name):
        """
        Get the method `name` of the current class.

        Look through all methods and interfaces of the current class to
        find a method called `name`. If this class inherits from another
        class, that class is also searched to ensure that the inherited
        methods are available.

        Parameters
        ----------
        name : str
            The name of the attribute we are looking for.

        Returns
        -------
        FunctionDef
            The definition of the method.

        Raises
        ------
        ValueError
            Raised if the method cannot be found.
        """
        try:
            method = next(i for i in chain(self.methods, self.interfaces) if i.name == name)
        except StopIteration:
            method = None
            i = 0
            n_classes = len(self.superclasses)
            while method is None and i<n_classes:
                try:
                    method = self.superclasses[i].get_method(name)
                except StopIteration:
                    method = None

        if method is None:
            errors.report(f"Can't find method {name} in class {self.name}",
                    severity='fatal', symbol=self)

        return method

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

    @property
    def is_unused(self):
        return False


class Import(Basic):

    """Represents inclusion of dependencies in the code.

    Parameters
    ----------
    source : str, DottedName, AsName
        the module from which we import
    target : str, AsName, list, tuple
        targets to import
    ignore_at_print : bool
        indicates whether the import should be printed
    mod : Module
        The module describing the source

    Examples
    --------
    >>> from pyccel.ast.core import Import
    >>> from pyccel.ast.core import DottedName
    >>> Import('foo')
    import foo

    >>> abc = DottedName('foo', 'bar', 'baz')
    >>> Import(abc)
    import foo.bar.baz

    >>> Import('foo', 'bar')
    from foo import bar
    """
    __slots__ = ('_source','_target','_ignore_at_print','_source_mod')
    _attribute_nodes = ()

    def __init__(self, source, target = None, ignore_at_print = False, mod = None):

        if not source is None:
            source = Import._format(source)

        self._source = source
        self._target = set()
        self._source_mod      = mod
        self._ignore_at_print = ignore_at_print
        if target is None:
            if pyccel_stage == "syntactic":
                target = []
            else:
                raise KeyError("Missing argument 'target'")
        elif not iterable(target):
            target = [target]
        if pyccel_stage == "syntactic":
            for i in target:
                self._target.add(Import._format(i))
        else:
            for i in target:
                assert isinstance(i, (AsName, Module))
                if isinstance(i, Module):
                    self._target.add(AsName(i,source))
                else:
                    self._target.add(i)
        super().__init__()

    @staticmethod
    def _format(i):
        if isinstance(i, str):
            if '.' in i:
                return DottedName(*i.split('.'))
            else:
                return PyccelSymbol(i)
        if isinstance(i, (DottedName, AsName, PyccelSymbol)):
            return i
        else:
            raise TypeError('Expecting a string, PyccelSymbol DottedName, given {}'.format(type(i)))

    @property
    def target(self):
        return self._target

    @property
    def source(self):
        return self._source

    @property
    def ignore(self):
        return self._ignore_at_print

    @ignore.setter
    def ignore(self, to_ignore):
        if not isinstance(to_ignore, bool):
            raise TypeError('to_ignore must be a boolean.')
        self._ignore_at_print = to_ignore

    def __str__(self):
        source = str(self.source)
        if len(self.target) == 0:
            return 'import {source}'.format(source=source)
        else:
            target = ', '.join([str(i) for i in self.target])
            return 'from {source} import {target}'.format(source=source,
                    target=target)

    def define_target(self, new_target):
        """
        Add an additional target to the imports
        I.e. if imp is an Import defined as:
        >>> from numpy import ones

        and we call imp.define_target('cos')
        then it becomes:
        >>> from numpy import ones, cos

        Parameter
        ---------
        new_target: str/AsName/iterable of str/AsName
                    The new import target
        """
        if iterable(new_target):
            self._target.update(new_target)
        else:
            self._target.add(new_target)

    def find_module_target(self, new_target):
        for t in self._target:
            if isinstance(t, AsName) and new_target == t.name:
                return t.target
            elif new_target == t:
                return t
        return None

    @property
    def source_module(self):
        """ The module describing the Import source
        """
        return self._source_mod


# TODO: Should Declare have an optional init value for each var?

class FuncAddressDeclare(Basic):

    """Represents a FunctionAddress declaration in the code.

    Parameters
    ----------
    variable:
        An instance of FunctionAddress.
    intent: None, str
        one among {'in', 'out', 'inout'}
    value: PyccelAstNode
        variable value
    static: bool
        True for a static declaration of an array.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, FunctionAddress, FuncAddressDeclare
    >>> x = Variable('float', 'x')
    >>> y = Variable('float', 'y')
    >>> FuncAddressDeclare(FunctionAddress('f', [x], [y], []))
    """
    __slots__ = ('_variable','_intent','_value','_static')
    _attribute_nodes = ('_variable', '_value')

    def __init__(
        self,
        variable,
        intent=None,
        value=None,
        static=False,
        ):

        if not isinstance(variable, FunctionAddress):
            raise TypeError('variable must be of type FunctionAddress, given {0}'.format(variable))

        if intent:
            if not intent in ['in', 'out', 'inout']:
                raise ValueError("intent must be one among {'in', 'out', 'inout'}")

        if not isinstance(static, bool):
            raise TypeError('Expecting a boolean for static attribute')

        self._variable  = variable
        self._intent    = intent
        self._value     = value
        self._static    = static
        super().__init__()

    @property
    def results(self):
        return self._variable.results

    @property
    def arguments(self):
        return self._variable.arguments

    @property
    def name(self):
        return self._variable.name

    @property
    def variable(self):
        return self._variable

    @property
    def intent(self):
        return self._intent

    @property
    def value(self):
        return self._value

    @property
    def static(self):
        return self._static

# ARA : issue-999 add is_external for external function exported through header files
class Declare(Basic):

    """Represents a variable declaration in the code.

    Parameters
    ----------
    dtype : DataType
        The type for the declaration.
    variable(s)
        A single variable or an iterable of Variables. If iterable, all
        Variables must be of the same type.
    intent: None, str
        one among {'in', 'out', 'inout'}
    value: PyccelAstNode
        variable value
    static: bool
        True for a static declaration of an array.
    external: bool
        True for a function declared through a header
    module_variable : bool
        True for a variable which belongs to a module

    Examples
    --------
    >>> from pyccel.ast.core import Declare, Variable
    >>> Declare('int', Variable('int', 'n'))
    Declare(NativeInteger(), (n,), None)
    >>> Declare('float', Variable('float', 'x'), intent='out')
    Declare(NativeFloat(), (x,), out)
    """
    __slots__ = ('_dtype','_variable','_intent','_value',
                 '_static','_passed_from_dotted', '_external',
                 '_module_variable')
    _attribute_nodes = ('_variable', '_value')

    def __init__(
        self,
        dtype,
        variable,
        intent=None,
        value=None,
        static=False,
        passed_from_dotted = False,
        external = False,
        module_variable = False
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
            raise TypeError('Expecting a boolean for static attribute')

        if not isinstance(passed_from_dotted, bool):
            raise TypeError('Expecting a boolean for passed_from_dotted attribute')

        if not isinstance(external, bool):
            raise TypeError('Expecting a boolean for external attribute')

        if not isinstance(module_variable, bool):
            raise TypeError('Expecting a boolean for module_variable attribute')

        self._dtype = dtype
        self._variable = variable
        self._intent = intent
        self._value = value
        self._static = static
        self._passed_from_dotted = passed_from_dotted
        self._external = external
        self._module_variable = module_variable
        super().__init__()

    @property
    def dtype(self):
        return self._dtype

    @property
    def variable(self):
        return self._variable

    @property
    def intent(self):
        return self._intent

    @property
    def value(self):
        return self._value

    @property
    def static(self):
        return self._static

    @property
    def passed_from_dotted(self):
        """ Argument is the lhs of a DottedFunction
        """
        return self._passed_from_dotted

    @property
    def external(self):
        return self._external

    @property
    def module_variable(self):
        """ Indicates whether the variable is scoped to
        a module
        """
        return self._module_variable

    def __repr__(self):
        return 'Declare({})'.format(repr(self.variable))

class Break(Basic):

    """Represents a break in the code."""
    __slots__ = ()
    _attribute_nodes = ()


class Continue(Basic):

    """Represents a continue in the code."""
    __slots__ = ()
    _attribute_nodes = ()


class Raise(Basic):

    """Represents a raise in the code."""
    __slots__ = ()
    _attribute_nodes = ()



class SymbolicPrint(Basic):

    """Represents a print function of symbolic expressions in the code.

    Parameters
    ----------
    expr : PyccelAstNode
        The expression to print

    Examples
    --------
    >>> from pyccel.ast.internals import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
    """
    __slots__ = ('_expr',)
    _attribute_nodes = ('_expr',)

    def __init__(self, expr):
        if not iterable(expr):
            raise TypeError('Expecting an iterable')

        for i in expr:
            if not isinstance(i, (Lambda, SymbolicAssign,
                              SympyFunction)):
                raise TypeError('Expecting Lambda, SymbolicAssign, SympyFunction for {}'.format(i))

        self._expr = expr

        super().__init__()

    @property
    def expr(self):
        return self._expr


class Del(Basic):

    """Represents a memory deallocation in the code.

    Parameters
    ----------
    variables : list, tuple
        a list of pyccel variables

    Examples
    --------
    >>> from pyccel.ast.core import Del, Variable
    >>> x = Variable('float', 'x', rank=2, shape=(10,2), memory_handling='heap')
    >>> Del([x])
    Del([x])
    """
    __slots__ = ('_variables',)
    _attribute_nodes = ('_variables',)

    def __init__(self, expr):

        # TODO: check that the variable is allocatable

        if not iterable(expr):
            expr = tuple([expr])

        self._variables = expr
        super().__init__()

    @property
    def variables(self):
        return self._variables


class EmptyNode(Basic):
    """
    Represents an empty node in the abstract syntax tree (AST).
    When a subtree is removed from the AST, we replace it with an EmptyNode
    object that acts as a placeholder. Using an EmptyNode instead of None
    is more explicit and avoids confusion. Further, finding a None in the AST
    is signal of an internal bug.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import EmptyNode
    >>> EmptyNode()

    """
    __slots__ = ()
    _attribute_nodes = ()

    def __str__(self):
        return ''


class Comment(Basic):

    """Represents a Comment in the code.

    Parameters
    ----------
    text : str
       the comment line

    Examples
    --------
    >>> from pyccel.ast.core import Comment
    >>> Comment('this is a comment')
    # this is a comment
    """
    __slots__ = ('_text')
    _attribute_nodes = ()

    def __init__(self, text):
        self._text = text
        super().__init__()

    @property
    def text(self):
        return self._text

    def __str__(self):
        return '# {0}'.format(str(self.text))

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments.
        """
        kwargs = dict(text = self.text)
        return (apply, (self.__class__, (), kwargs))


class SeparatorComment(Comment):

    """Represents a Separator Comment in the code.

    Parameters
    ----------
    mark : str
        marker

    Examples
    --------
    >>> from pyccel.ast.core import SeparatorComment
    >>> SeparatorComment(n=40)
    # ........................................
    """
    __slots__ = ()

    def __init__(self, n):
        text = """.""" * n
        super().__init__(text)

class AnnotatedComment(Basic):

    """Represents a Annotated Comment in the code.

    Parameters
    ----------
    accel : str
       accelerator id. One among {'acc'}

    txt: str
        statement to print

    Examples
    --------
    >>> from pyccel.ast.core import AnnotatedComment
    >>> AnnotatedComment('acc', 'parallel')
    AnnotatedComment(acc, parallel)
    """
    __slots__ = ('_accel','_txt')
    _attribute_nodes = ()

    def __init__(self, accel, txt):
        self._accel = accel
        self._txt = txt
        super().__init__()

    @property
    def accel(self):
        return self._accel

    @property
    def txt(self):
        return self._txt

    def __getnewargs__(self):
        """used for Pickling self."""

        args = (self.accel, self.txt)
        return args

class CommentBlock(Basic):

    """ Represents a Block of Comments

    Parameters
    ----------
    txt : str

    """
    __slots__ = ('_header','_comments')
    _attribute_nodes = ()

    def __init__(self, txt, header = 'CommentBlock'):
        if not isinstance(txt, str):
            raise TypeError('txt must be of type str')
        txt = txt.replace('"','')
        txts = txt.split('\n')

        self._header = header
        self._comments = txts

        super().__init__()

    @property
    def comments(self):
        return self._comments

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, header):
        self._header = header


class Assert(Basic):
    """
    Represents an assert statement in the code.

    Parameters
    ----------
    test: PyccelAstNode
        boolean expression to check
    """
    __slots__ = ('_test',)
    _attribute_nodes = ('_test',)

    def __init__(self, test):
        if pyccel_stage != 'syntactic':
            if test.dtype is not NativeBool():
                test = PythonBool(test)
        self._test = test
        super().__init__()

    @property
    def test(self):
        return self._test


class Pass(Basic):

    """Basic class for pass instruction."""
    __slots__ = ()
    _attribute_nodes = ()

class Exit(Basic):

    """Basic class for exits."""
    __slots__ = ()
    _attribute_nodes = ()

#TODO: [EB 26.01.2021] Do we need this unused class?
class ErrorExit(Exit):

    """Exit with error."""
    __slots__ = ()

class IfSection(Basic):
    """Represents a condition and associated code block
    in an if statement in the code.

    Parameters
    ----------
    cond : PyccelAstNode
           A boolean expression indicating whether or not the block
           should be executed
    body : CodeBlock
           The code to be executed in the condition is satisfied

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign, IfSection, CodeBlock
    >>> n = PyccelSymbol('n')
    >>> IfSection((n>1), CodeBlock([Assign(n,n-1)]))
    IfSection((n>1), CodeBlock([Assign(n,n-1)]))
    """
    __slots__ = ('_condition','_block')
    _attribute_nodes = ('_condition','_block')

    def __init__(self, cond, body):

        if pyccel_stage == 'semantic' and cond.dtype is not NativeBool():
            cond = PythonBool(cond)
        if isinstance(body, (list, tuple)):
            body = CodeBlock(body)
        elif isinstance(body, CodeBlock):
            body = body
        else:
            raise TypeError('body is not iterable or CodeBlock')

        self._condition = cond
        self._block     = body

        super().__init__()

    @property
    def condition(self):
        return self._condition

    @property
    def body(self):
        return self._block

    def __iter__(self):
        return iter((self.condition, self.body))

    def __str__(self):
        return "IfSec({},{})".format(str(self.condition), str(self.body))

class If(Basic):

    """Represents a if statement in the code.

    Parameters
    ----------
    args : IfSection
           All arguments are sections of the complete If block

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> from pyccel.ast.core import Assign, If
    >>> n = PyccelSymbol('n')
    >>> i1 = IfSection((n>1), [Assign(n,n-1)])
    >>> i2 = IfSection(True, [Assign(n,n+1)])
    >>> If(i1, i2)
    If(IfSection((n>1), [Assign(n,n-1)]), IfSection(True, [Assign(n,n+1)]))
    """
    __slots__ = ('_blocks',)
    _attribute_nodes = ('_blocks',)

    # TODO add type check in the semantic stage

    def __init__(self, *args):

        if not all(isinstance(a, IfSection) for a in args):
            raise TypeError("An If must be composed of IfSections")

        self._blocks = args

        super().__init__()

    @property
    def blocks(self):
        return self._blocks

    @property
    def bodies(self):
        return [b.body for b in self._blocks]

    def __str__(self):
        return "If({})".format(','.join(str(b) for b in self.blocks))

class StarredArguments(Basic):
    __slots__ = ('_starred_obj',)
    _attribute_nodes = ('_starred_obj',)
    def __init__(self, args):
        self._starred_obj = args
        super().__init__()

    @property
    def args_var(self):
        return self._starred_obj

# ...

class InProgram(PyccelAstNode):
    """
    Class representing the boolean:
    __name__ == '__main__'
    """
    _dtype = NativeBool()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _attribute_nodes = ()
    __slots__ = ()

# ...

class Decorator(Basic):
    """ Class representing a function decorator.
    For now this is just designed to handle the pyccel decorators

    Parameters
    ----------
    name : str
            The name of the decorator
    """
    __slots__ = ('_name',)
    _attribute_nodes = ()

    def __init__(self, name):
        self._name = name
        super().__init__()

    @property
    def name(self):
        """ Return the name of the decorator
        """
        return self._name

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

        if isinstance(cls_base, PythonRange):
            if not isinstance(it.name, DottedName):
                raise TypeError('Expecting a DottedName, given  {0}'.format(type(it.name)))

            args = []
            for i in [cls_base.start, cls_base.stop, cls_base.step]:
                if isinstance(i, Variable):
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

            return [PythonRange(*args)]

    elif isinstance(it, ConstructorCall):
        cls_base = it.this.cls_base

        # arguments[0] is 'self'

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
                    if isinstance(a_new, (IndexedElement, str, Variable)):
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
        elif isinstance(expr, sp_And):
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
                    if isinstance(a_new, (IndexedElement, str, Variable)):
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


    if not len(ends) == len(starts):
        raise ValueError('wrong number of starts/ends')

    # ...

    return [PythonRange(s, e, 1) for (s, e) in zip(starts, ends)]

#==============================================================================

