from collections.abc import Iterable
from sympy import Symbol, Tuple
from sympy.core.function      import Function, Application
from sympy.core.expr          import Expr, AtomicExpr
from sympy.logic.boolalg      import Boolean as sp_Boolean
from sympy.tensor             import IndexedBase
from sympy.matrices.matrices  import MatrixBase
from sympy.utilities.misc     import filldedent
from sympy.utilities.iterables          import iterable
from sympy.tensor.array.ndim_array      import NDimArray

from .basic     import Basic, PyccelAstNode
from .datatypes import (datatype, DataType, CustomDataType,
                        NativeInteger, NativeBool, NativeReal,
                        NativeComplex, NativeRange, NativeString,
                        NativeGeneric, default_precision)
from .literals       import LiteralInteger, Nil
from .operators import PyccelMinus, PyccelOperator

__all__ = (
    'DottedName',
    'DottedVariable',
    'IndexedElement',
    'IndexedVariable',
    'PyccelArraySize',
    'Slice',
    'TupleVariable',
    'ValuedVariable',
    'Variable',
    'VariableAddress'
)

class Slice(Basic, PyccelOperator):

    """Represents a slice in the code.

    Parameters
    ----------
    start : Symbol or int
        starting index

    stop : Symbol or int
        ending index

    step : Symbol or int default None

    Examples
    --------
    >>> from sympy import symbols
    >>> from pyccel.ast.core import Slice
    >>> start, end, step = symbols('start, stop, step', integer=True)
    >>> Slice(start, stop)
    start : stop
    >>> Slice(None, stop)
     : stop
    >>> Slice(start, None)
    start :
    >>> Slice(start, stop, step)
    start : stop : step
    """

    def __new__(cls, start, stop, step = None):
        return Basic.__new__(cls, start, stop, step)

    def __init__(self, start, stop, step = None):
        self._start = start
        self._stop = stop
        self._step = step
        if self.stage == 'syntactic':
                return
        if start is not None and not (hasattr(start, 'dtype') and isinstance(start.dtype, NativeInteger)):
            raise TypeError('Slice start must be Integer or None')
        if stop is not None and not (hasattr(stop, 'dtype') and isinstance(stop.dtype, NativeInteger)):
            raise TypeError('Slice stop must be Integer or None')
        if step is not None and not (hasattr(step, 'dtype') and isinstance(step.dtype, NativeInteger)):
            raise TypeError('Slice step must be Integer or None')

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.start is None:
            start = ''
        else:
            start = sstr(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = sstr(self.stop)
        return '{0} : {1}'.format(start, stop)

    def __str__(self):
        if self.start is None:
            start = ''
        else:
            start = str(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = str(self.stop)
        return '{0} : {1}'.format(start, stop)

class Variable(Symbol, PyccelAstNode):

    """Represents a typed variable.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType,
        or a str (bool, int, real).

    name : str, list, DottedName
        The sympy object the variable represents. This can be either a string
        or a dotted name, when using a Class attribute.

    rank : int
        used for arrays. [Default value: 0]

    allocatable: bool
        used for arrays, if we need to allocate memory [Default value: False]

    is_stack_array: bool
        used for arrays, if memory should be allocated on the stack [Default value: False]

    is_pointer: bool
        if object is a pointer [Default value: False]

    is_target: bool
        if object is pointed to by another variable [Default value: False]

    is_polymorphic: bool
        if object can be instance of class or any inherited class [Default value: False]

    is_optional: bool
        if object is an optional argument of a function [Default value: False]

    shape: int or list
        shape of the array. [Default value: None]

    cls_base: class
        class base if variable is an object or an object member [Default value: None]

    order : str
        used for arrays. Indicates whether the data is stored in C or Fortran format in memory [Default value: 'C']

    precision : str
        Precision of the data type [Default value: depends on the datatype]

    is_argument: bool
        if object is the argument of a function [Default value: False]

    is_kwonly: bool
        if object is an argument which can only be specified using its keyword

    is_const: bool
        if object is a const argument of a function [Default value: False]

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> Variable('int', 'n')
    n
    >>> n = 4
    >>> Variable('real', 'x', rank=2, shape=(n,2), allocatable=True)
    x
    >>> Variable('int', DottedName('matrix', 'n_rows'))
    matrix.n_rows
    """

    def __new__( cls, *args, **kwargs ):
        return Basic.__new__(cls)

    def __init__(
        self,
        dtype,
        name,
        *,
        rank=0,
        allocatable=False,
        is_stack_array = False,
        is_pointer=False,
        is_const=False,
        is_target=False,
        is_polymorphic=None,
        is_optional=False,
        shape=None,
        cls_base=None,
        order='C',
        precision=0,
        is_argument=False,
        is_kwonly=False,
        allows_negative_indexes=False
        ):

        # ------------ PyccelAstNode Properties ---------------
        if isinstance(dtype, str) or str(dtype) == '*':

            dtype = datatype(str(dtype))
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(rank, int):
            raise TypeError('rank must be an instance of int.')

        if rank == 0:
            shape = ()

        if shape is None:
            shape = tuple(None for i in range(rank))

        if not precision:
            if isinstance(dtype, NativeInteger):
                precision = default_precision['int']
            elif isinstance(dtype, NativeReal):
                precision = default_precision['real']
            elif isinstance(dtype, NativeComplex):
                precision = default_precision['complex']
            elif isinstance(dtype, NativeBool):
                precision = default_precision['bool']
        if not isinstance(precision,int) and precision is not None:
            raise TypeError('precision must be an integer or None.')

        self._alloc_shape = shape
        self._dtype = dtype
        self._shape = self.process_shape(shape)
        self._rank  = rank
        self._precision = precision

        # ------------ Variable Properties ---------------
        # if class attribute
        if isinstance(name, str):
            name = name.split(""".""")
            if len(name) == 1:
                name = name[0]
            else:
                name = DottedName(*name)

        if not isinstance(name, (str, DottedName)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(name)))
        self._name = name

        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self.allocatable = allocatable

        if not isinstance(is_const, bool):
            raise TypeError('is_const must be a boolean.')
        self.is_const = is_const

        if not isinstance(is_stack_array, bool):
            raise TypeError('is_stack_array must be a boolean.')
        self._is_stack_array = is_stack_array

        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self.is_pointer = is_pointer

        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self.is_target = is_target

        if is_polymorphic is None:
            if isinstance(dtype, CustomDataType):
                is_polymorphic = dtype.is_polymorphic
            else:
                is_polymorphic = False
        elif not isinstance(is_polymorphic, bool):
            raise TypeError('is_polymorphic must be a boolean.')
        self._is_polymorphic = is_polymorphic

        if not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')
        self._is_optional = is_optional

        if not isinstance(allows_negative_indexes, bool):
            raise TypeError('allows_negative_indexes must be a boolean.')
        self._allows_negative_indexes = allows_negative_indexes

        self._cls_base       = cls_base
        self._order          = order
        self._is_argument    = is_argument
        self._is_kwonly      = is_kwonly

    def process_shape(self, shape):
        if not hasattr(shape,'__iter__'):
            shape = [shape]

        new_shape = []
        for i,s in enumerate(shape):
            if isinstance(s,(LiteralInteger, PyccelArraySize)):
                new_shape.append(s)
            elif isinstance(s, int):
                new_shape.append(LiteralInteger(s))
            elif s is None or isinstance(s,(Variable, Slice, PyccelAstNode, Function)):
                new_shape.append(PyccelArraySize(self, i))
            else:
                raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: Integer(pyccel),'
                                'Variable, Slice, PyccelAstNode, Integer(sympy), int, Function')
        return tuple(new_shape)

    @property
    def name(self):
        return self._name

    @property
    def alloc_shape(self):
        return self._alloc_shape

    @property
    def allocatable(self):
        return self._allocatable

    @allocatable.setter
    def allocatable(self, allocatable):
        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self._allocatable = allocatable

    @property
    def cls_base(self):
        return self._cls_base

    @property
    def is_pointer(self):
        return self._is_pointer

    @is_pointer.setter
    def is_pointer(self, is_pointer):
        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self._is_pointer = is_pointer

    @property
    def is_target(self):
        return self._is_target

    @is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target

    @property
    def is_polymorphic(self):
        return self._is_polymorphic

    @property
    def is_optional(self):
        return self._is_optional

    @property
    def order(self):
        return self._order

    @property
    def is_stack_array(self):
        return self._is_stack_array

    @is_stack_array.setter
    def is_stack_array(self, is_stack_array):
        self._is_stack_array = is_stack_array

    @property
    def allows_negative_indexes(self):
        return self._allows_negative_indexes

    @allows_negative_indexes.setter
    def allows_negative_indexes(self, allows_negative_indexes):
        self._allows_negative_indexes = allows_negative_indexes

    @property
    def is_argument(self):
        return self._is_argument

    @property
    def is_kwonly(self):
        return self._is_kwonly

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
        print( '  precision      = {}'.format(self.precision))
        print( '  rank           = {}'.format(self.rank))
        print( '  order          = {}'.format(self.order))
        print( '  allocatable    = {}'.format(self.allocatable))
        print( '  shape          = {}'.format(self.shape))
        print( '  cls_base       = {}'.format(self.cls_base))
        print( '  is_pointer     = {}'.format(self.is_pointer))
        print( '  is_target      = {}'.format(self.is_target))
        print( '  is_polymorphic = {}'.format(self.is_polymorphic))
        print( '  is_optional    = {}'.format(self.is_optional))
        print( '<<<')

    def clone(self, name, new_class = None, **kwargs):

        # TODO check it is up to date

        if (new_class is None):
            cls = self.__class__
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
            )
    def rename(self, newname):
        """Change variable name."""

        self._name = newname

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
        args = (
            self.dtype,
            self.name)
        kwargs = {
            'rank' : self.rank,
            'allocatable': self.allocatable,
            'is_pointer':self.is_pointer,
            'is_polymorphic':self.is_polymorphic,
            'is_optional':self.is_optional,
            'shape':self.shape,
            'cls_base':self.cls_base,
            }

        out =  (lambda f,a,k: f(*a, **k), (Variable, args, kwargs))
        return out

    def _eval_subs(self, old, new):
        return self

    def _eval_is_positive(self):
        #we do this inorder to infere the type of Pow expression correctly
        return self.is_real

class DottedName(Basic):

    """
    Represents a dotted variable.

    Examples
    --------
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

class DottedVariable(AtomicExpr, sp_Boolean, PyccelAstNode):

    """
    Represents a dotted variable.
    """

    def __new__(cls, lhs, rhs):

        if PyccelAstNode.stage != 'syntactic':
            lhs_cls = lhs.__class__.__name__
            rhs_cls = rhs.__class__.__name__
            if not isinstance(lhs, (
                Variable,
                IndexedVariable,
                IndexedElement,
                IndexedBase,
                Indexed,
                DottedVariable,
                )):
                raise TypeError('Expecting a Variable or a function call, got instead {0} of type {1}'.format(str(lhs),
                                str(type(lhs))))

            if rhs_cls not in (
                'Variable',
                'IndexedVariable',
                'IndexedElement',
                'IndexedBase',
                'Indexed',
                'FunctionCall',
                'Function',
                'TupleVariable',
                ):
                raise TypeError('Expecting a Variable or a function call, got instead {0} of type {1}'.format(str(rhs),
                                str(type(rhs))))

        return Basic.__new__(cls, lhs, rhs)

    def __init__(self, lhs, rhs):
        if self.stage == 'syntactic':
            return
        self._dtype     = rhs.dtype
        self._rank      = rhs.rank
        self._precision = rhs.precision
        self._shape     = rhs.shape
        self._order     = rhs.order

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def allocatable(self):
        return self._args[1].allocatable

    @allocatable.setter
    def allocatable(self, allocatable):
        self._args[1].allocatable = allocatable

    @property
    def is_pointer(self):
        return self._args[1].is_pointer

    @is_pointer.setter
    def is_pointer(self, is_pointer):
        self._args[1].is_pointer = is_pointer

    @property
    def is_target(self):
        return self._args[1].is_target

    @is_target.setter
    def is_target(self, is_target):
        self._args[1].is_target = is_target

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

    Parameters
    ----------
    variable: Variable
        A single variable
    value: Variable, or instance of Native types
        value associated to the variable

    Examples
    --------
    >>> from pyccel.ast.core import ValuedVariable
    >>> n  = ValuedVariable('int', 'n', value=4)
    >>> n
    n := 4
    """

    def __new__(cls, *args, **kwargs):

        # we remove value from kwargs,
        # since it is not a valid argument for Variable

        kwargs.pop('value', Nil())

        return Variable.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):

        # if value is not given, we set it to Nil
        self._value = kwargs.pop('value', Nil())
        Variable.__init__(self, *args, **kwargs)

    @property
    def value(self):
        return self._value

    def _sympystr(self, printer):
        sstr = printer.doprint

        name = sstr(self.name)
        value = sstr(self.value)
        return '{0}={1}'.format(name, value)

class TupleVariable(Variable):

    """Represents a tuple variable in the code.

    Parameters
    ----------
    arg_vars: Iterable
        Multiple variables contained within the tuple

    Examples
    --------
    >>> from pyccel.ast.core import TupleVariable, Variable
    >>> v1 = Variable('int','v1')
    >>> v2 = Variable('bool','v2')
    >>> n  = TupleVariable([v1, v2],'n')
    >>> n
    n
    """

    def __new__(cls, arg_vars, dtype, name, *args, **kwargs):

        # if value is not given, we set it to Nil
        # we also remove value from kwargs,
        # since it is not a valid argument for Variable

        return Variable.__new__(cls, dtype, name, *args, **kwargs)

    def __init__(self, arg_vars, dtype, name, *args, **kwargs):
        self._vars = tuple(arg_vars)
        self._inconsistent_shape = not all(arg_vars[0].shape==a.shape   for a in arg_vars[1:])
        self._is_homogeneous = not dtype is NativeGeneric()
        Variable.__init__(self, dtype, name, *args, **kwargs)

    def get_vars(self):
        if self._is_homogeneous:
            indexed_var = IndexedVariable(self, dtype=self.dtype, shape=self.shape,
                prec=self.precision, order=self.order, rank=self. rank)
            args = [Slice(None,None)]*(self.rank-1)
            return [indexed_var[[i] + args] for i in range(len(self._vars))]
        else:
            return self._vars

    def get_var(self, variable_idx):
        return self._vars[variable_idx]

    def rename_var(self, variable_idx, new_name):
        self._vars[variable_idx].rename(new_name)

    def __getitem__(self,idx):
        if isinstance(idx, LiteralInteger):
            idx = idx.p
        return self.get_var(idx)

    def __iter__(self):
        return self._vars.__iter__()

    def __len__(self):
        return len(self._vars)

    @property
    def inconsistent_shape(self):
        return self._inconsistent_shape

    @property
    def is_homogeneous(self):
        return self._is_homogeneous

    @is_homogeneous.setter
    def is_homogeneous(self, is_homogeneous):
        self._is_homogeneous = is_homogeneous

    @Variable.allocatable.setter
    def allocatable(self, allocatable):
        if not isinstance(allocatable, bool):
            raise TypeError('allocatable must be a boolean.')
        self._allocatable = allocatable
        for var in self._vars:
            var.allocatable = allocatable

    @Variable.is_pointer.setter
    def is_pointer(self, is_pointer):
        if not isinstance(is_pointer, bool):
            raise TypeError('is_pointer must be a boolean.')
        self._is_pointer = is_pointer
        for var in self._vars:
            var.is_pointer = is_pointer

    @Variable.is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target
        for var in self._vars:
            var.is_target = is_target

class Constant(ValuedVariable, PyccelAstNode):

    """

    Examples
    --------

    """



class IndexedVariable(IndexedBase, PyccelAstNode):

    """
    Represents an indexed variable, like x in x[i], in the code.

    Examples
    --------
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

        if isinstance(label, Application):
            label_name = type(label)
        else:
            label_name = str(label)

        return IndexedBase.__new__(cls, label_name, shape=shape)

    def __init__(
        self,
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


        self._dtype      = dtype
        self._precision  = prec
        self._rank       = rank
        self._order      = order
        kw_args['order'] = order
        self._kw_args    = kw_args
        self._label      = label

    def __getitem__(self, *args):

        if len(args) == 1 and isinstance(args[0], (Tuple, tuple, list)):
            args = args[0]

        if self.shape and len(self.shape) != len(args):
            raise IndexError('Rank mismatch.')

        obj = IndexedElement(self, *args)
        return obj

    @property
    def order(self):
        return self.kw_args['order']

    @property
    def kw_args(self):
        return self._kw_args

    @property
    def name(self):
        return self._args[0]

    @property
    def internal_variable(self):
        return self._label


    def clone(self, name):
        cls = self.__class__
        # TODO what about kw_args in __new__?
        return cls(name, shape=self.shape, dtype=self.dtype,
                   prec=self.precision, order=self.order, rank=self.rank)

    def _eval_subs(self, old, new):
        return self

    def __str__(self):
        return str(self.name)


class IndexedElement(Expr, PyccelAstNode):

    """
    Represents a mathematical object with indices.

    Examples
    --------
    >>> from sympy import symbols, Idx
    >>> from pyccel.ast.core import IndexedVariable, IndexedElement
    >>> i, j = symbols('i j', cls=Idx)
    >>> A = IndexedVariable('A', dtype='int')
    >>> IndexedElement(A, i, j)
    IndexedElement(A, i, j)
    >>> IndexedElement(A, i, j) == A[i, j]
    True
    """

    def __new__(
        cls,
        base,
        *args,
        **kw_args
        ):

        if not args:
            raise IndexError('Indexed needs at least one index.')
        if isinstance(base, (str, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base,
                IndexedBase):
            raise TypeError(filldedent("""
                Indexed expects string, Symbol, or IndexedBase as base."""))

        if isinstance(base, (NDimArray, Iterable, Tuple,
                      MatrixBase)) and all([i.is_number for i in args]):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]
        return Expr.__new__(cls, base, *args, **kw_args)

    def __init__(
        self,
        base,
        *args,
        **kw_args
        ):

        self._label = self._args[0]
        self._indices = self._args[1:]
        dtype = self.base.dtype
        shape = self.base.shape
        rank  = self.base.rank
        order = self.base.order
        self._precision = self.base.precision
        if isinstance(dtype, NativeInteger):
            self._dtype = NativeInteger()
        elif isinstance(dtype, NativeReal):
            self._dtype = NativeReal()
        elif isinstance(dtype, NativeComplex):
            self._dtype = NativeComplex()
        elif isinstance(dtype, NativeBool):
            self._dtype = NativeBool()
        elif isinstance(dtype, NativeString):
            self._dtype = NativeString()
        elif not isinstance(dtype, NativeRange):
            raise TypeError('Undefined datatype')

        if shape is not None:
            new_shape = []
            for a,s in zip(args, shape):
                if isinstance(a, Slice):
                    start = a.start
                    stop   = a.stop
                    stop   = s if stop is None else stop
                    if start is None:
                        new_shape.append(stop)
                    else:
                        new_shape.append(PyccelMinus(stop, start))
            self._shape = tuple(new_shape)
            self._rank  = len(new_shape)
        else:
            new_rank = rank
            for i in range(rank):
                if not isinstance(args[i], Slice):
                    new_rank -= 1
            self._rank = new_rank
        self._order = order

    @property
    def base(self):
        return self._label

    @property
    def indices(self):
        return self._indices
class VariableAddress(Basic, PyccelAstNode):

    """Represents the address of a variable.
    E.g. In C
    VariableAddress(Variable('int','a'))                     is  &a
    VariableAddress(Variable('int','a', is_pointer=True))    is   a
    """

    def __init__(self, variable):
        if not isinstance(variable, Variable):
            raise TypeError('variable must be a variable')
        self._variable = variable

        self._shape     = variable.shape
        self._rank      = variable.rank
        self._dtype     = variable.dtype
        self._precision = variable.precision
        self._order     = variable.order

    @property
    def variable(self):
        return self._variable

class PyccelArraySize(Function, PyccelAstNode):
    def __new__(cls, arg, index):
        is_PyccelAstNode = isinstance(arg, PyccelAstNode) and \
                (arg.shape is None or all(a.shape is None for a in arg.shape))
        if not (is_PyccelAstNode or isinstance(arg, Variable) or hasattr(arg, '__len__')):
            raise TypeError('Uknown type of  %s.' % type(arg))

        return Basic.__new__(cls, arg, index)

    def __init__(self, arg, index):
        self._dtype = NativeInteger()
        self._rank  = 0
        self._shape = ()
        self._precision = default_precision['integer']

    @property
    def arg(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def _sympystr(self, printer):
        return 'Shape({},{})'.format(str(self.arg), str(self.index))
