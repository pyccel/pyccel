# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" This module contains all classes which are used to handle memory block labels at
different stages of pyccel. Memory block labels are usually either Variables or Indexed
variables
"""
import inspect

from pyccel.errors.errors   import Errors
from pyccel.utilities.stage import PyccelStage

from .basic     import PyccelAstNode, TypedAstNode
from .datatypes import PyccelType, InhomogeneousTupleType, HomogeneousListType, HomogeneousSetType, DictType
from .datatypes import ContainerType, HomogeneousTupleType, CharType, StringType
from .internals import PyccelArrayShapeElement, Slice, PyccelSymbol
from .literals  import LiteralInteger, Nil, LiteralEllipsis
from .operators import (PyccelMinus, PyccelDiv, PyccelMul,
                        PyccelUnarySub, PyccelAdd)
from .numpytypes import NumpyNDArrayType

errors = Errors()
pyccel_stage = PyccelStage()

__all__ = (
    'AnnotatedPyccelSymbol',
    'Constant',
    'DottedName',
    'DottedVariable',
    'IndexedElement',
    'TupleVariable',
    'Variable'
)

class Variable(TypedAstNode):
    """
    Represents a typed variable.

    Represents a variable in the code and stores all useful properties which allow
    for easy usage of this variable.

    Parameters
    ----------
    class_type : PyccelType
        The Python type of the variable.

    name : str, list, DottedName
        The name of the variable represented. This can be either a string
        or a dotted name, when using a Class attribute.

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
        each of the dimensions of an array. The elements of the tuple should be None or TypedAstNodes.

    cls_base : class, default: None
        Class base if variable is an object or an object member.

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
    >>> from pyccel.ast.datatypes import PythonNativeInt, PythonNativeFloat
    >>> from pyccel.ast.core import Variable
    >>> Variable(PythonNativeInt(), 'n')
    n
    >>> n = 4
    >>> Variable(PythonNativeFloat(), 'x', shape=(n,2), memory_handling='heap')
    x
    >>> Variable(PythonNativeInt(), DottedName('matrix', 'n_rows'))
    matrix.n_rows
    """
    __slots__ = ('_name', '_alloc_shape', '_memory_handling', '_is_const', '_is_target',
            '_is_optional', '_allows_negative_indexes', '_cls_base', '_is_argument', '_is_temp',
            '_shape','_is_private','_class_type')
    _attribute_nodes = ()

    def __init__(
        self,
        class_type,
        name,
        *,
        memory_handling='stack',
        is_const=False,
        is_target=False,
        is_optional=False,
        is_private=False,
        shape=None,
        cls_base=None,
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
                name = PyccelSymbol(name[0])
            else:
                name = DottedName(*name)

        if name == '':
            raise ValueError("Variable name can't be empty")

        assert isinstance(name, (PyccelSymbol, DottedName))
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
        self._is_argument    = is_argument
        self._is_temp        = is_temp

        # ------------ TypedAstNode Properties ---------------
        assert isinstance(class_type, PyccelType)
        rank = class_type.rank

        if rank == 0:
            assert shape is None

        elif shape is None:
            shape = tuple(None for i in range(class_type.container_rank))

        self._alloc_shape = shape
        self._class_type = class_type
        self._shape = self.process_shape(shape)

        # Ignore codegen stage due to #861
        assert pyccel_stage == 'codegen' or class_type.shape_is_compatible(self._shape)

    def process_shape(self, shape):
        """
        Simplify the provided shape and ensure it has the expected format.

        The provided shape is the shape used to create the object, and it can
        be a long expression. In most cases where the shape is required the
        provided shape is inconvenient, or it might have become invalid. This
        function therefore replaces those expressions with calls to the function
        `PyccelArrayShapeElement`.

        Parameters
        ----------
        shape : iterable of int
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
            elif s is None or isinstance(s, TypedAstNode):
                new_shape.append(PyccelArrayShapeElement(self, LiteralInteger(i)))
            else:
                raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: LiteralInteger,'
                                'Variable, Slice, TypedAstNode, int, Function')
        return tuple(new_shape)

    def shape_can_change(self, i):
        """
        Indicate if the shape can change in the i-th dimension.

        Indicate whether the Variable's shape can change in the i-th dimension
        at run time.

        Parameters
        ----------
        i : int
            The dimension over which the shape can change at runtime.

        Returns
        -------
        bool
            Whether or not the variable shape can change in the i-th dimension.
        """
        return (self.is_alias and not isinstance(self.class_type, InhomogeneousTupleType)) or \
                isinstance(self.class_type, (HomogeneousListType, HomogeneousSetType, DictType))

    def set_changeable_shape(self):
        """
        Indicate that the Variable's shape is unknown at compilation time.

        Indicate that the exact shape is unknown, e.g. if the allocate is done in
        an If block.
        """
        self._shape = tuple(PyccelArrayShapeElement(self, LiteralInteger(i)) for i in range(self.rank))
        self._alloc_shape = tuple(PyccelArrayShapeElement(self, LiteralInteger(i)) for i in range(self.rank))

    def set_init_shape(self, shape):
        """
        Set the shape that was passed to the variable upon creation.

        Set the shape that was passed to the variable upon creation. Normally this can be
        deduced when the variable was created, however this may not be the case if the
        variable was predeclared via a header or an annotation.

        Parameters
        ----------
        shape : tuple
            The shape of the array. A tuple whose elements indicate the number of elements along
            each of the dimensions of an array. The elements of the tuple should be None or TypedAstNodes.
        """
        self._alloc_shape = shape
        self._shape = self.process_shape(shape)

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
    def cls_base(self):
        """ Class from which the Variable inherits
        """
        return self._cls_base

    @property
    def is_const(self):
        """
        Indicates whether the Variable is constant within its context.

        Indicates whether the Variable is constant within its context.
        True if the Variable is constant, false if it can be modified.
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
        """
        User friendly method to check if the variable is a numpy.ndarray.

        User friendly method to check if the variable is an ndarray.
        """
        return isinstance(self.class_type, NumpyNDArrayType)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f'{type(self).__name__}({self.name}, type={repr(self.class_type)})'

    def __eq__(self, other):
        if type(self) is type(other):
            return self._name == other.name
        return False

    def __hash__(self):
        return hash((type(self).__name__, self._name))

    def inspect(self):
        """
        Print a short summary of the Variable and its parameters.

        Print a short summary of the Variable and its parameters.
        This function is useful for debugging.
        """

        print('>>> Variable')
        print(f'  name               = {self.name}')
        print(f'  type               = {self.class_type}')
        print(f'  memory_handling    = {self.memory_handling}')
        print(f'  shape              = {self.shape}')
        print(f'  cls_base           = {self.cls_base}')
        print(f'  is_target          = {self.is_target}')
        print(f'  is_optional        = {self.is_optional}')
        print( '<<<')

    def clone(self, name, new_class = None, **kwargs):
        """
        Create a clone of the current variable.

        Create a new Variable object of the chosen class
        with the provided name and options. All non-specified
        options will match the current instance.

        Parameters
        ----------
        name : str
            The name of the new Variable.
        new_class : type, optional
            The class type of the new Variable (e.g. DottedVariable).
            The default is the same class type.
        **kwargs : dict
            Dictionary containing any keyword-value
            pairs which are valid constructor keywords.

        Returns
        -------
        Variable
            The cloned variable.
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
        if 'shape' not in kwargs:
            new_kwargs['shape'] = self.alloc_shape

        return cls(**new_kwargs)

    def rename(self, newname):
        """ Forbidden method for renaming the variable
        """
        # The name is part of the hash so it must never change
        raise RuntimeError('Cannot modify hash definition')

    def __getitem__(self, *args):

        if self.rank < len(args):
            raise IndexError('Rank mismatch.')

        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, (tuple, list)):
                args = arg0
            elif isinstance(arg0, int):
                self_len = self.shape[0]
                if isinstance(self_len, LiteralInteger) and arg0 >= int(self_len):
                    raise StopIteration

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

class DottedName(PyccelAstNode):

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
    _attribute_nodes = ('_name',)

    def __new__(cls, *args):
        if len(args) == 1:
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, *args):

        self._name = args
        super().__init__()

    @property
    def name(self):
        """ The different components of the name
        (these were separated by dots)
        """
        return self._name

    def __repr__(self):
        return """.""".join(str(n) for n in self.name)

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return str(self) != str(other)

    def __hash__(self):
        return hash(str(self))

class Constant(Variable):
    """
    Class for expressing constant values (e.g. pi).

    Class for expressing constant values (e.g. pi).

    Parameters
    ----------
    *args : tuple
        See pyccel.ast.variable.Variable.

    value : bool|int|float|complex
        The value that the constant represents.

    **kwargs : dict
        See pyccel.ast.variable.Variable.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeFloat
    >>> from pyccel.ast.variable import Constant
    >>> import math
    >>> Constant(PythonNativeFloat(), 'pi' , value=math.pi )
    Constant('pi', type=NativeFloat())
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
        return f'{self.name}={self.value}'



class IndexedElement(TypedAstNode):
    """
    Represents an indexed object in the code.

    Represents an object which is a subset of a base object. The
    indexed object is retrieved by passing indices to the base
    object using the `[]` syntax.

    In the semantic stage, the base object is an array, tuple or
    list. This function then determines the new rank and shape of
    the data block.

    In the syntactic stage, this object is more versatile, it
    stores anything which is indexed using `[]` syntax. This can
    additionally include classes, maps, etc.

    Parameters
    ----------
    base : Variable | PyccelSymbol | DottedName
        The object being indexed.

    *indices : tuple of TypedAstNode
        The values used to index the base.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, IndexedElement
    >>> from pyccel.ast.datatypes import PythonNativeInt
    >>> A = Variable(PythonNativeInt(), 'A', shape=(2,3), rank=2)
    >>> i = Variable(PythonNativeInt(), 'i')
    >>> j = Variable(PythonNativeInt(), 'j')
    >>> IndexedElement(A, (i, j))
    IndexedElement(A, i, j)
    >>> IndexedElement(A, i, j) == A[i, j]
    True
    """
    __slots__ = ('_label', '_indices','_shape','_class_type', '_is_slice')
    _attribute_nodes = ('_label', '_indices', '_shape')

    def __init__(self, base, *indices):

        if not indices:
            raise IndexError('Indexed needs at least one index.')

        self._label = base
        self._shape = None
        if pyccel_stage == 'syntactic':
            self._indices = indices
            super().__init__()
            return

        shape = base.shape
        rank  = base.class_type.container_rank
        assert len(indices) <= rank

        if any(not isinstance(a, (int, TypedAstNode, Slice, LiteralEllipsis)) for a in indices):
            errors.report("Index is not of valid type",
                    symbol = indices, severity = 'fatal')

        if len(indices) == 1 and isinstance(indices[0], LiteralEllipsis):
            self._indices = tuple(LiteralInteger(a) if isinstance(a, int) else a for a in indices)
            indices = [Slice(None,None)]*rank
        # Add empty slices to fully index the object
        elif len(indices) < rank:
            indices = indices + tuple([Slice(None, None)]*(rank-len(indices)))
            self._indices = tuple(LiteralInteger(a) if isinstance(a, int) else a for a in indices)
        else:
            self._indices = tuple(LiteralInteger(a) if isinstance(a, int) else a for a in indices)

        def make_positive(idx, shape):
            """
            Use the shape to convert a literal negative index to a positive index.
            """
            if isinstance(idx, Slice):
                return Slice(make_positive(idx.start, shape),
                             make_positive(idx.stop, shape),
                             idx.step)
            else:
                try:
                    if int(idx) < 0:
                        return PyccelAdd(shape, idx, simplify=True)
                except TypeError:
                    pass
                return idx

        self._indices = tuple(make_positive(idx, shape[i]) for i, idx in enumerate(self._indices))

        if isinstance(base.class_type, InhomogeneousTupleType):
            assert len(self._indices) == 1 and isinstance(self._indices[0], LiteralInteger)
            self._class_type = base.class_type[self._indices[0]]
            self._is_slice = False

        else:
            # Calculate new shape
            new_shape = []
            from .mathext import MathCeil

            negative_idxs_possible = getattr(base, 'allows_negative_indexes', False)

            for a,s in zip(indices, shape):
                if isinstance(a, Slice):
                    start = a.start
                    stop  = a.stop if a.stop is not None else s
                    step  = a.step
                    if negative_idxs_possible:
                        try:
                            start_int = int(start)
                            stop_int = int(stop)
                        except TypeError:
                            new_shape.append(None)
                            continue

                    _shape = stop if start is None else PyccelMinus(stop, start, simplify=True)
                    if step is not None:
                        try:
                            if int(step) < 0:
                                start = s if a.start is None else start
                                _shape = start if a.stop is None else PyccelMinus(start, stop, simplify=True)
                                step = PyccelUnarySub(step, simplify=True)
                        except TypeError as e:
                            if negative_idxs_possible:
                                new_shape.append(None)
                                continue

                        _shape = MathCeil(PyccelDiv(_shape, step, simplify=True))
                    new_shape.append(_shape)
            if isinstance(base.class_type, HomogeneousTupleType):
                new_shape.extend(shape[1:])
            new_rank = len(new_shape)

            if new_rank == 0:
                self._class_type = base.class_type.element_type
                self._is_slice = False
            elif isinstance(base.class_type, HomogeneousTupleType):
                class_type = base.class_type
                rank = base.rank
                while new_rank != rank:
                    class_type = class_type.element_type
                    rank -= 1
                self._class_type = class_type
                self._shape = tuple(new_shape)
                self._is_slice = False
            else:
                self._class_type = base.class_type.switch_rank(new_rank) if new_rank != rank \
                                    else base.class_type
                self._is_slice = True
                self._shape = tuple(new_shape)

            if isinstance(self._class_type, CharType):
                self._class_type = StringType()
                self._shape = (1,)

        if isinstance(self._class_type, InhomogeneousTupleType) and self._shape is None:
            self._shape = (LiteralInteger(len(self._class_type)),)

        super().__init__()

        # This must appear after the call to super().__init__ to avoid setting "self"  as a
        # user of itself
        if isinstance(self._class_type, ContainerType) and self._shape is None:
            self._shape = tuple(PyccelArrayShapeElement(self, i) \
                                for i in range(self._class_type.container_rank))

        # Ignore codegen stage due to #861
        assert pyccel_stage == 'codegen' or self._class_type.shape_is_compatible(self._shape)

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
        indices = ','.join(str(i) for i in self.indices)
        return f'{self.base}[{indices}]'

    def __repr__(self):
        indices = ','.join(repr(i) for i in self.indices)
        return f'{repr(self.base)}[{indices}]'

    def __getitem__(self, *args):

        if self.class_type.container_rank < len(args):
            raise IndexError('Rank mismatch.')

        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]

        if self._is_slice:
            new_indexes = []
            j = 0
            base = self.base
            for i in self.indices:
                if isinstance(i, Slice) and j<len(args):
                    current_arg = args[j]
                    if isinstance(current_arg, Slice):
                        raise NotImplementedError("Can't extract a slice from a slice")
                    else:
                        if i.step == 1 or i.step is None:
                            incr = current_arg
                        else:
                            incr = PyccelMul(i.step, current_arg, simplify = True)
                        if i.start != 0 and i.start is not None:
                            incr = PyccelAdd(i.start, incr, simplify = True)
                    i = incr
                    j += 1
                new_indexes.append(i)
            return IndexedElement(base, *new_indexes)
        else:
            return IndexedElement(self, *args)

    @property
    def is_const(self):
        """
        Indicates whether the Variable is constant within its context.

        Indicates whether the Variable is constant within its context.
        True if the Variable is constant, false if it can be modified.
        """
        return self.base.is_const

    @property
    def allows_negative_indexes(self):
        """
        Indicate whether variables used to index this Variable can be negative.

        Indicate whether variables used to index this Variable can be negative.
        """
        return self.base.allows_negative_indexes

    @property
    def is_slice(self):
        """
        Indicates whether this instance represents a slice.

        Indicates whether this instance represents a slice or an element.
        """
        return self._is_slice

    def __hash__(self):
        return hash((self.base, self._indices))

    def __eq__(self, other):
        if isinstance(other, IndexedElement):
            return self.base == other.base and self.indices == other.indices
        else:
            return False

class DottedVariable(Variable):
    """
    Class representing a dotted variable.

    Represents a dotted variable. This is usually
    a variable which is a member of a class

    E.g.
    a = AClass()
    a.b = 3

    In this case b is a DottedVariable where the lhs is a.

    Parameters
    ----------
    *args : tuple
        See pyccel.ast.variable.Variable.

    lhs : Variable
        The Variable on the right of the '.'.

    **kwargs : dict
        See pyccel.ast.variable.Variable.
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
        class_type = repr(self.class_type)
        classname = type(self).__name__
        return f'{classname}({lhs}.{name}, type={class_type})'

class AnnotatedPyccelSymbol(PyccelAstNode):
    """
    Class representing a symbol in the code which has an annotation.

    Symbolic placeholder for a Python variable, which has a name but no type yet.
    This is very generic, and it can also represent a function or a module.

    Parameters
    ----------
    name : str
        Name of the symbol.

    annotation : PyccelAstNode, optional
        The annotation describing the type that the object will have.
        This should be an object from the type_annotations or typingext module
        (e.g. SyntacticTypeAnnotation, FunctionTypeAnnotation, TypingFinal).
        The annotation may be None if the argument is a bound class argument
        whose type can be auto-deduced later.

    is_temp : bool
        Indicates if the symbol is a temporary object. This either means that the
        symbol represents an object originally named `_` in the code, or that the
        symbol represents an object created by Pyccel in order to assign a
        temporary object. This is sometimes necessary to facilitate the translation.
    """
    __slots__ = ('_name', '_annotation')
    _attribute_nodes = ('_name',)

    def __init__(self, name, annotation, is_temp = False):
        assert annotation is None or isinstance(annotation, PyccelAstNode)
        if isinstance(name, (PyccelSymbol, DottedName)):
            self._name = name
        elif isinstance(name, str):
            self._name = PyccelSymbol(name, is_temp)
        else:
            raise TypeError(f"Name should be a string or a PyccelSymbol not a {type(name)}")
        self._annotation = annotation
        super().__init__()

    @property
    def name(self):
        """
        Get the PyccelSymbol describing the name.

        Get the PyccelSymbol describing the name of the symbol in the code.
        """
        return self._name

    @property
    def annotation(self):
        """
        Get the annotation.

        Get the annotation left on the symbol. This should be a type annotation.
        """
        return self._annotation

    def __str__(self):
        return f'{self.name} : {self.annotation}'

