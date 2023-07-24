# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" This module contains all classes which are used to handle memory block labels at
different stages of pyccel. Memory block labels are usually either Variables or Indexed
variables
"""
import inspect

from pyccel.errors.errors   import Errors
from pyccel.utilities.stage import PyccelStage

from .basic     import Basic, PyccelAstNode
from .datatypes import (datatype, DataType,
                        NativeInteger, NativeBool, NativeFloat,
                        NativeComplex)
from .internals import PyccelArrayShapeElement, Slice, get_final_precision
from .literals  import LiteralInteger, Nil
from .operators import (PyccelMinus, PyccelDiv, PyccelMul,
                        PyccelUnarySub, PyccelAdd)

errors = Errors()
pyccel_stage = PyccelStage()

__all__ = (
    'Constant',
    'DottedName',
    'DottedVariable',
    'HomogeneousTupleVariable',
    'IndexedElement',
    'InhomogeneousTupleVariable',
    'TupleVariable',
    'Variable'
)

class Variable(PyccelAstNode):
    """
    Represents a typed variable.

    Represents a variable in the code and stores all useful properties which allow
    for easy usage of this variable.

    Parameters
    ----------
    dtype : str, DataType
        The type of the variable. Can be either a DataType,
        or a str (bool, int, float).

    name : str, list, DottedName
        The name of the variable represented. This can be either a string
        or a dotted name, when using a Class attribute.

    rank : int, default: 0
        The number of dimensions for an array.

    memory_handling : str, default: 'stack'
        'heap' is used for arrays, if we need to allocate memory on the heap.
        'stack' if memory should be allocated on the stack, represents stack arrays and scalars.
        'alias' if object allows access to memory stored in another variable.

    is_const : bool, default: False
        Indicates if object is a const argument of a function.

    is_target : bool, default: False
        Indicates if object is pointed to by another variable.

    is_optional : bool, default: False
        Indicates if object is an optional argument of a function.

    is_private : bool, default: False
        Indicates if object is private within a Module.

    shape : tuple, default: None
        The shape of the array. A tuple whose elements indicate the number of elements along
        each of the dimensions of an array. The elements of the tuple should be None or PyccelAstNodes.

    cls_base : class, default: None
        Class base if variable is an object or an object member.

    order : str, default: 'C'
        Used for arrays. Indicates whether the data is stored in C or Fortran format in memory.
        See order_docs.md in the developer docs for more details.

    precision : str, default: 0
        Precision of the data type.

    is_argument : bool, default: False
        Indicates if object is the argument of a function.

    is_temp : bool, default: False
        Indicates if this symbol represents a temporary variable created by Pyccel,
        and was not present in the original Python code.

    allows_negative_indexes : bool, default: False
        Indicates if non-literal negative indexes should be correctly handled when indexing this
        variable. The default is False for performance reasons.

    Examples
    --------
    >>> from pyccel.ast.core import Variable
    >>> Variable('int', 'n')
    n
    >>> n = 4
    >>> Variable('float', 'x', rank=2, shape=(n,2), memory_handling='heap')
    x
    >>> Variable('int', DottedName('matrix', 'n_rows'))
    matrix.n_rows
    """
    __slots__ = ('_name', '_alloc_shape', '_memory_handling', '_is_const',
            '_is_target', '_is_optional', '_allows_negative_indexes',
            '_cls_base', '_is_argument', '_is_temp','_dtype','_precision',
            '_rank','_shape','_order','_is_private')
    _attribute_nodes = ()

    def __init__(
        self,
        dtype,
        name,
        *,
        rank=0,
        memory_handling='stack',
        is_const=False,
        is_target=False,
        is_optional=False,
        is_private=False,
        shape=None,
        cls_base=None,
        order=None,
        precision=0,
        is_argument=False,
        is_temp =False,
        allows_negative_indexes=False
        ):
        super().__init__()

        # ------------ Variable Properties ---------------
        # if class attribute
        if isinstance(name, str):
            name = name.split(""".""")
            if len(name) == 1:
                name = name[0]
            else:
                name = DottedName(*name)

        if name == '':
            raise ValueError("Variable name can't be empty")

        if not isinstance(name, (str, DottedName)):
            raise TypeError('Expecting a string or DottedName, given {0}'.format(type(name)))
        self._name = name

        if memory_handling not in ('heap', 'stack', 'alias'):
            raise ValueError("memory_handling must be 'heap', 'stack' or 'alias'")
        self._memory_handling = memory_handling

        if not isinstance(is_const, bool):
            raise TypeError('is_const must be a boolean.')
        self._is_const = is_const

        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self.is_target = is_target

        if not isinstance(is_optional, bool):
            raise TypeError('is_optional must be a boolean.')
        self._is_optional = is_optional

        if not isinstance(is_private, bool):
            raise TypeError('is_private must be a boolean.')
        self._is_private = is_private

        if not isinstance(allows_negative_indexes, bool):
            raise TypeError('allows_negative_indexes must be a boolean.')
        self._allows_negative_indexes = allows_negative_indexes

        self._cls_base       = cls_base
        self._order          = order
        self._is_argument    = is_argument
        self._is_temp        = is_temp

        # ------------ PyccelAstNode Properties ---------------
        if isinstance(dtype, str) or str(dtype) == '*':

            dtype = datatype(str(dtype))
        elif not isinstance(dtype, DataType):
            raise TypeError('datatype must be an instance of DataType.')

        if not isinstance(rank, int):
            raise TypeError('rank must be an instance of int.')

        if rank == 0:
            assert shape is None
            assert order is None

        elif shape is None:
            shape = tuple(None for i in range(rank))
        else:
            assert len(shape) == rank

        if rank == 1:
            assert order is None
        elif rank > 1:
            assert order in ('C', 'F')

        if not precision:
            if isinstance(dtype, (NativeInteger, NativeFloat, NativeComplex, NativeBool)):
                precision = -1
        if not isinstance(precision,int) and precision is not None:
            raise TypeError('precision must be an integer or None.')

        self._alloc_shape = shape
        self._dtype = dtype
        self._rank  = rank
        self._shape = self.process_shape(shape)
        self._precision = precision
        if self._rank < 2:
            self._order = None

    def process_shape(self, shape):
        """
        Simplify the provided shape and ensure it has the expected format.

        The provided shape is the shape used to create the object. In most
        cases where the shape is required we do not require this expression
        (which can be quite long). This function therefore replaces those
        expressions with calls to PyccelArrayShapeElement.

        Parameters
        ----------
        shape : iterable of integers
            The array shape to be simplified.

        Returns
        -------
        tuple
            The simplified array shape.

        """
        if self.rank == 0:
            return None
        elif not hasattr(shape,'__iter__'):
            shape = [shape]

        new_shape = []
        for i, s in enumerate(shape):
            if self.shape_can_change(i):
                # Shape of a pointer can change
                new_shape.append(PyccelArrayShapeElement(self, LiteralInteger(i)))
            elif isinstance(s, LiteralInteger):
                new_shape.append(s)
            elif isinstance(s, int):
                new_shape.append(LiteralInteger(s))
            elif s is None or isinstance(s, PyccelAstNode):
                new_shape.append(PyccelArrayShapeElement(self, LiteralInteger(i)))
            else:
                raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: LiteralInteger,'
                                'Variable, Slice, PyccelAstNode, int, Function')
        return tuple(new_shape)

    def shape_can_change(self, i):
        """
        Indicates if the shape can change in the i-th dimension.
        """
        return self.is_alias

    def set_changeable_shape(self):
        """
        Indicate that the Variable's shape is unknown at compilation time.

        Indicate that the exact shape is unknown, e.g. if the allocate is done in
        an If block.

        """
        self._shape = [PyccelArrayShapeElement(self, LiteralInteger(i)) for i in range(self.rank)]

    @property
    def name(self):
        """ Name of the variable
        """
        return self._name

    @property
    def alloc_shape(self):
        """ Shape of the variable at allocation

        The shape used in pyccel is usually simplified to contain
        only Literals and PyccelArraySizes but the shape for
        the allocation of x cannot be `Shape(x)`
        """
        return self._alloc_shape

    @property
    def memory_handling(self):
        """ Indicates whether a Variable has a dynamic size
        """
        return self._memory_handling

    @memory_handling.setter
    def memory_handling(self, memory_handling):
        if memory_handling not in ('heap', 'stack', 'alias'):
            raise ValueError("memory_handling must be 'heap', 'stack' or 'alias'")
        self._memory_handling = memory_handling

    @property
    def is_alias(self):
        """ Indicates if variable is an alias
        """
        return self.memory_handling == 'alias'

    @property
    def on_heap(self):
        """ Indicates if memory is allocated on the heap
        """
        return self.memory_handling == 'heap'

    @property
    def on_stack(self):
        """ Indicates if memory is allocated on the stack
        """
        return self.memory_handling == 'stack'

    @property
    def is_stack_array(self):
        """ Indicates if the variable is located on stack and is an array
        """
        return self.on_stack and self.rank > 0

    @property
    def is_stack_scalar(self):
        """ Indicates if the variable is located on stack and is a scalar
        """
        return self.on_stack and self.rank == 0

    @property
    def cls_base(self):
        """ Class from which the Variable inherits
        """
        return self._cls_base

    @property
    def is_const(self):
        """ Indicates if the Variable is constant
        within its context
        """
        return self._is_const

    @property
    def is_temp(self):
        """
        Indicates if this symbol represents a temporary variable created by Pyccel,
		and was not present in the original Python code [default value : False].
        """
        return self._is_temp

    @property
    def is_target(self):
        """ Indicates if the data in this Variable is
        shared with (pointed at by) another Variable
        """
        return self._is_target

    @is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target

    @property
    def is_optional(self):
        """ Indicates if the Variable is optional
        in this context
        """
        return self._is_optional

    @property
    def is_private(self):
        """ Indicates if the Variable is private
        within the Module
        """
        return self._is_private

    @property
    def allows_negative_indexes(self):
        """ Indicates whether variables used to
        index this Variable can be negative
        """
        return self._allows_negative_indexes

    @allows_negative_indexes.setter
    def allows_negative_indexes(self, allows_negative_indexes):
        self._allows_negative_indexes = allows_negative_indexes

    @property
    def is_argument(self):
        """ Indicates whether the Variable is
        a function argument in this context
        """
        return self._is_argument

    def declare_as_argument(self):
        """
        Indicate that the variable is used as an argument.

        This function is called by FunctionDefArgument to ensure that
        arguments are correctly flagged as such.
        """
        self._is_argument = True

    @property
    def is_ndarray(self):
        """user friendly method to check if the variable is an ndarray:
            1. have a rank > 0
            2. dtype is one among {int, bool, float, complex}
        """

        if self.rank == 0:
            return False
        return isinstance(self.dtype, (NativeInteger, NativeBool,
                          NativeFloat, NativeComplex))

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return '{}({}, dtype={})'.format(type(self).__name__, repr(self.name), repr(self.dtype))

    def __eq__(self, other):
        if type(self) is type(other):
            return self._name == other.name
        return False

    def __hash__(self):
        return hash((type(self).__name__, self._name))

    def inspect(self):
        """inspects the variable."""

        print('>>> Variable')
        print( '  name               = {}'.format(self.name))
        print( '  dtype              = {}'.format(self.dtype))
        print( '  precision          = {}'.format(get_final_precision(self)))
        print( '  rank               = {}'.format(self.rank))
        print( '  order              = {}'.format(self.order))
        print( '  memory_handling    = {}'.format(self.memory_handling))
        print( '  shape              = {}'.format(self.shape))
        print( '  cls_base           = {}'.format(self.cls_base))
        print( '  is_target          = {}'.format(self.is_target))
        print( '  is_optional        = {}'.format(self.is_optional))
        print( '<<<')

    def use_exact_precision(self):
        """
        Change precision from default python precision to
        equivalent numpy precision
        """
        if not self._is_argument:
            self._precision = get_final_precision(self)

    def clone(self, name, new_class = None, **kwargs):
        """
        Create a new Variable object of the chosen class
        with the provided name and options

        Parameters
        ==========
        name      : str
                    The name of the new Variable
        new_class : type
                    The class of the new Variable
                    The default is the same class
        kwargs    : dict
                    Dictionary containing any keyword-value
                    pairs which are valid constructor keywords
        """

        if (new_class is None):
            cls = self.__class__
        else:
            cls = new_class

        args = inspect.signature(Variable.__init__)
        new_kwargs = {k:getattr(self, '_'+k) \
                            for k in args.parameters.keys() \
                            if '_'+k in dir(self)}
        new_kwargs.update(kwargs)
        new_kwargs['name'] = name

        return cls(**new_kwargs)

    def rename(self, newname):
        """ Forbidden method for renaming the variable
        """
        # The name is part of the hash so it must never change
        raise RuntimeError('Cannot modify hash definition')

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
            'memory_handling': self.memory_handling,
            'is_optional':self.is_optional,
            'shape':self.shape,
            'cls_base':self.cls_base,
            }

        apply = lambda func, args, kwargs: func(*args, **kwargs)
        out =  (apply, (Variable, args, kwargs))
        return out

    def __getitem__(self, *args):

        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]

        if self.rank < len(args):
            raise IndexError('Rank mismatch.')

        return IndexedElement(self, *args)

    def invalidate_node(self):
        # Don't invalidate Variables
        pass

    @is_temp.setter
    def is_temp(self, is_temp):
        if not isinstance(is_temp, bool):
            raise TypeError("is_temp must be a boolean")
        elif is_temp:
            raise ValueError("Variables cannot become temporary")
        self._is_temp = is_temp

class DottedName(Basic):

    """
    Represents a dotted object.

    Represents an object accessed via a dot. This usually means that
    the object belongs to a class or module.

    Parameters
    ----------
    *args : tuple of PyccelSymbol
        The different symbols making up the dotted name.

    Examples
    --------
    >>> from pyccel.ast.core import DottedName
    >>> DottedName('matrix', 'n_rows')
    matrix.n_rows
    >>> DottedName('pyccel', 'stdlib', 'parallel')
    pyccel.stdlib.parallel
    """
    __slots__ = ('_name',)
    _attribute_nodes = ()

    def __init__(self, *args):

        self._name = args
        super().__init__()

    @property
    def name(self):
        """ The different components of the name
        (these were separated by dots)
        """
        return self._name

    def __str__(self):
        return """.""".join(str(n) for n in self.name)

    def __repr__(self):
        return """.""".join(repr(n) for n in self.name)

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return str(self) != str(other)

    def __hash__(self):
        return hash(str(self))

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
    __slots__ = ()

    @property
    def is_ndarray(self):
        return False

class HomogeneousTupleVariable(TupleVariable):

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
    __slots__ = ()
    is_homogeneous = True

    def shape_can_change(self, i):
        """
        Indicates if the shape can change in the i-th dimension
        """
        return self.is_alias and i == (self.rank-1)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        assert isinstance(self.shape[0], LiteralInteger)
        return (self[i] for i in range(self.shape[0]))

class InhomogeneousTupleVariable(TupleVariable):

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
    __slots__ = ('_vars',)
    _attribute_nodes = ('_vars',)
    is_homogeneous = False

    def __init__(self, arg_vars, dtype, name, *args, **kwargs):
        self._vars = tuple(arg_vars)
        super().__init__(dtype, name, *args, **kwargs)

    def get_vars(self):
        """ Get the variables saved internally in the tuple
        (used for inhomogeneous variables)
        """
        return self._vars

    def get_var(self, variable_idx):
        """ Get the n-th variable saved internally in the
        tuple (used for inhomogeneous variables)

        Parameters
        ==========
        variable_idx : int/LiteralInteger
                       The index of the variable which we
                       wish to collect
        """
        return self._vars[variable_idx]

    def rename_var(self, variable_idx, new_name):
        """ Rename the n-th variable saved internally in the
        tuple (used for inhomogeneous variables)

        Parameters
        ==========
        variable_idx : int/LiteralInteger
                       The index of the variable which we
                       wish to collect
        new_name     : str
                       The new name of the variable
        """
        self._vars[variable_idx].rename(new_name)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            sub_idx = idx[1:]
            idx = idx[0]
        else:
            sub_idx = []

        var = self.get_var(idx)

        if len(sub_idx) > 0:
            return var[sub_idx]
        else:
            return var

    def __iter__(self):
        return self._vars.__iter__()

    def __len__(self):
        return len(self._vars)

    @Variable.memory_handling.setter
    def memory_handling(self, memory_handling):
        if memory_handling not in ('heap', 'stack', 'alias'):
            raise ValueError("memory_handling must be 'heap', 'stack' or 'alias'")
        self._memory_handling = memory_handling
        for var in self._vars:
            if var.rank > 0:
                var.memory_handling = memory_handling

    @Variable.is_target.setter
    def is_target(self, is_target):
        if not isinstance(is_target, bool):
            raise TypeError('is_target must be a boolean.')
        self._is_target = is_target
        for var in self._vars:
            if var.rank > 0:
                var.is_target = is_target

class Constant(Variable):

    """
    Class for expressing constant values (e.g. pi)

    Parameters
    ----------
    *args, **kwargs : See pyccel.ast.variable.Variable

    value : Type matching dtype
            The value that the constant represents

    Examples
    --------
    >>> from pyccel.ast.variable import Constant
    >>> import math
    >>> Constant('float', 'pi' , value=math.pi )
    Constant('pi', dtype=NativeFloat())

    """
    __slots__ = ('_value',)
    # The value of a constant is not a translated object
    _attribute_nodes = ()

    def __init__(self, *args, value = Nil(), **kwargs):
        self._value = value
        super().__init__(*args, **kwargs)

    @property
    def value(self):
        """ Immutable value of the constant
        """
        return self._value

    def __str__(self):
        name = str(self.name)
        value = str(self.value)
        return '{0}={1}'.format(name, value)



class IndexedElement(PyccelAstNode):

    """
    Represents a mathematical object with indices.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, IndexedElement
    >>> A = Variable('A', dtype='int', shape=(2,3), rank=2)
    >>> i = Variable('i', dtype='int')
    >>> j = Variable('j', dtype='int')
    >>> IndexedElement(A, i, j)
    IndexedElement(A, i, j)
    >>> IndexedElement(A, i, j) == A[i, j]
    True
    """
    __slots__ = ('_label', '_indices','_dtype','_precision','_shape','_rank','_order')
    _attribute_nodes = ('_label', '_indices')

    def __init__(
        self,
        base,
        *args,
        **kw_args
        ):

        if not args:
            raise IndexError('Indexed needs at least one index.')

        self._label = base

        if pyccel_stage == 'syntactic':
            self._indices = args
            super().__init__()
            return

        self._dtype = base.dtype
        self._precision = base.precision

        shape = base.shape
        rank  = base.rank

        # Add empty slices to fully index the object
        if len(args) < rank:
            args = args + tuple([Slice(None, None)]*(rank-len(args)))

        if any(not isinstance(a, (int, PyccelAstNode, Slice)) for a in args):
            errors.report("Index is not of valid type",
                    symbol = args, severity = 'fatal')

        self._indices = tuple(LiteralInteger(a) if isinstance(a, int) else a for a in args)
        super().__init__()

        # Calculate new shape

        if shape is not None:
            new_shape = []
            from .mathext import MathCeil
            for a,s in zip(args, shape):
                if isinstance(a, Slice):
                    start = a.start
                    stop  = a.stop if a.stop is not None else s
                    step  = a.step
                    if isinstance(start, PyccelUnarySub):
                        start = PyccelAdd(s, start, simplify=True)
                    if isinstance(stop, PyccelUnarySub):
                        stop = PyccelAdd(s, stop, simplify=True)

                    _shape = stop if start is None else PyccelMinus(stop, start, simplify=True)
                    if step is not None:
                        if isinstance(step, PyccelUnarySub):
                            start = s if a.start is None else start
                            _shape = start if a.stop is None else PyccelMinus(start, stop, simplify=True)
                            step = PyccelUnarySub(step)

                        _shape = MathCeil(PyccelDiv(_shape, step, simplify=True))
                    new_shape.append(_shape)
            self._rank  = len(new_shape)
            self._shape = None if self._rank == 0 else tuple(new_shape)
        else:
            new_rank = rank
            for i in range(rank):
                if not isinstance(args[i], Slice):
                    new_rank -= 1
            self._rank = new_rank

        self._order = None if self.rank < 2 else base.order

    @property
    def base(self):
        """ The object which is indexed
        """
        return self._label

    @property
    def indices(self):
        """ A tuple of indices used to index the variable
        """
        return self._indices

    def __str__(self):
        return '{}[{}]'.format(self.base, ','.join(str(i) for i in self.indices))

    def __repr__(self):
        return '{}[{}]'.format(repr(self.base), ','.join(repr(i) for i in self.indices))

    def __getitem__(self, *args):

        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]

        if self.rank < len(args):
            raise IndexError('Rank mismatch.')

        new_indexes = []
        j = 0
        base = self.base
        for i in self.indices:
            if isinstance(i, Slice) and j<len(args):
                if i.step == 1 or i.step is None:
                    incr = args[j]
                else:
                    incr = PyccelMul(i.step, args[j], simplify = True)
                if i.start != 0 and i.start is not None:
                    incr = PyccelAdd(i.start, incr, simplify = True)
                i = incr
                j += 1
            new_indexes.append(i)
        return IndexedElement(base, *new_indexes)

class DottedVariable(Variable):

    """
    Represents a dotted variable. This is usually
    a variable which is a member of a class

    E.g.
    a = AClass()
    a.b = 3

    In this case b is a DottedVariable where the lhs
    is a
    """
    __slots__ = ('_lhs',)
    _attribute_nodes = ('_lhs',)

    def __init__(self, *args, lhs, **kwargs):
        self._lhs = lhs
        super().__init__(*args, **kwargs)

    @property
    def lhs(self):
        """ The object before the final dot in the
        dotted variable

        e.g. for the DottedVariable:
        a.b
        The lhs is a
        """
        return self._lhs

    def __eq__(self, other):
        if type(self) is type(other):
            return self.name == other.name and self.lhs == other.lhs

        return False

    def __hash__(self):
        return hash((type(self).__name__, self.name, self.lhs))

    def __str__(self):
        return str(self.lhs)+'.'+str(self.name)

    def __repr__(self):
        lhs = repr(self.lhs)
        name = str(self.name)
        dtype = repr(self.dtype)
        classname = type(self).__name__
        return f'{classname}({lhs}.{name}, dtype={dtype}'
