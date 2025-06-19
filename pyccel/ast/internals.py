# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
File containing basic classes which are used throughout pyccel.
To avoid circular imports this file should only import from basic, datatypes, and literals
"""

from pyccel.utilities.stage import PyccelStage

from .basic     import PyccelAstNode, TypedAstNode, Immutable
from .datatypes import PythonNativeInt, PrimitiveIntegerType, SymbolicType
from .literals  import LiteralInteger

pyccel_stage = PyccelStage()

__all__ = (
    'Iterable',
    'PrecomputedCode',
    'PyccelArrayShapeElement',
    'PyccelArraySize',
    'PyccelFunction',
    'PyccelSymbol',
    'Slice',
)


class PyccelFunction(TypedAstNode):
    """
    Abstract class for function calls translated to Pyccel objects.

    A subclass of this base class represents calls to a specific internal
    function of Pyccel, which may be simplified at a later stage, or made
    available in the target language when printing the generated code.

    Parameters
    ----------
    *args : iterable
        The arguments passed to the function call.
    """
    __slots__ = ('_args',)
    _attribute_nodes = ('_args',)
    name = None

    def __init__(self, *args):
        self._args = tuple(args)
        super().__init__()

    @property
    def args(self):
        """
        The arguments passed to the function.

        Tuple containing all the arguments passed to the function call.
        """
        return self._args

    @property
    def is_elemental(self):
        """
        Whether the function acts elementwise on an array argument.

        Boolean indicating whether the (scalar) function should be called
        elementwise on an array argument. Here we set the default to False.
        """
        return False

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

class PyccelArraySize(PyccelFunction):
    """
    Gets the total number of elements in an array.

    Class representing a call to a function which would return
    the total number of elements in a multi-dimensional array.

    Parameters
    ----------
    arg : TypedAstNode
        An array of unknown size.
    """
    __slots__ = ()
    name = 'size'

    _shape = None
    _class_type = PythonNativeInt()

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def arg(self):
        """
        Object whose size is investigated.

        The argument of the function call, i.e. the object whose size is
        investigated.
        """
        return self._args[0]

    def __str__(self):
        return f'Size({self.arg})'

    def __eq__(self, other):
        if isinstance(other, PyccelArraySize):
            return self.arg == other.arg
        else:
            return False


class PyccelArrayShapeElement(PyccelFunction):
    """
    Gets the number of elements in a given dimension of an array.

    Class representing a call to a function which would return
    the shape of a multi-dimensional array in a given dimension.

    Parameters
    ----------
    arg : TypedAstNode
        An array of unknown shape.

    index : int
        The dimension along which the shape should be provided.
    """
    __slots__ = ()
    name = 'shape'

    _shape = None
    _class_type = PythonNativeInt()

    def __init__(self, arg, index):
        if not isinstance(arg, TypedAstNode):
            raise TypeError(f'Unknown type {type(arg)} of {arg}.')

        if isinstance(index, int):
            index = LiteralInteger(index)
        elif not isinstance(index, TypedAstNode):
            raise TypeError(f'Unknown type {type(index)} of {index}.')

        super().__init__(arg, index)

    @property
    def arg(self):
        """
        Object whose size is investigated.

        The first argument of the function call, i.e. the array whose size is
        investigated.
        """
        return self._args[0]

    @property
    def index(self):
        """
        Dimension along which the size is calculated.

        The second argument of the function call, i.e. the dimension along
        which the array size is calculated.
        """
        return self._args[1]

    def __repr__(self):
        return f'Shape({self.arg}, {self.index})'

    def __eq__(self, other):
        if isinstance(other, PyccelArrayShapeElement):
            return self.arg == other.arg and self.index == other.index
        else:
            return False


class Slice(PyccelAstNode):
    """
    Represents a slice in the code.

    An object of this class represents the slicing of a Numpy array along one of
    its dimensions. In most cases this corresponds to a Python slice in the user
    code, where it is represented by a `python.ast.Slice` object.

    In addition, at the wrapper and code generation stages, an integer index
    `i` used to create a view of a Numpy array is converted to an object
    `Slice(i, i+1, 1, slice_type = Slice.Element)`. This allows using C
    variadic arguments in the function `array_slicing` (in file
    pyccel/stdlib/ndarrays/ndarrays.c).

    Parameters
    ----------
    start : PyccelSymbol or int
        Starting index.

    stop : PyccelSymbol or int
        Ending index.

    step : PyccelSymbol or int, default=None
        The step between indices.

    slice_type : LiteralInteger
        The type of the slice. Either Slice.Range or Slice.Element.

    Examples
    --------
    >>> from pyccel.ast.internals import Slice, symbols
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
    __slots__ = ('_start','_stop','_step', '_slice_type')
    _attribute_nodes = ('_start','_stop','_step', '_slice_type')

    Range = LiteralInteger(1)
    Element = LiteralInteger(0)

    def __init__(self, start, stop, step = None, slice_type = Range):
        self._start = start
        self._stop = stop
        self._step = step
        self._slice_type = slice_type
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        assert start is None or isinstance(getattr(start.dtype, 'primitive_type', None), PrimitiveIntegerType)
        assert stop is None or isinstance(getattr(stop.dtype, 'primitive_type', None), PrimitiveIntegerType)
        assert step is None or isinstance(getattr(step.dtype, 'primitive_type', None), PrimitiveIntegerType)
        if slice_type not in (Slice.Range, Slice.Element):
            raise TypeError('Slice type must be Range (1) or Element (0)')

    @property
    def start(self):
        """ Index where the slicing of the object starts
        """
        return self._start

    @property
    def stop(self):
        """ Index until which the slicing takes place
        """
        return self._stop

    @property
    def step(self):
        """ The difference between each index of the
        objects in the slice
        """
        return self._step

    @property
    def slice_type(self):
        """ The type of the slice (Range or Element)
        Range <=> [..., :, ...]
        Element <=> [..., 3, ...]
        """
        return self._slice_type

    def __str__(self):
        if self.start is None:
            start = ''
        else:
            start = str(self.start)
        if self.stop is None:
            stop = ''
        else:
            stop = str(self.stop)
        return f'{start} : {stop} : {self.step}'


class PyccelSymbol(str, Immutable):
    """
    Class representing a symbol in the code.

    Symbolic placeholder for a Python variable, which has a name but no type yet.
    This is very generic, and it can also represent a function or a module.

    Parameters
    ----------
    name : str
        Name of the symbol.

    is_temp : bool
        Indicates if the symbol is a temporary object. This either means that the
        symbol represents an object originally named `_` in the code, or that the
        symbol represents an object created by Pyccel in order to assign a
        temporary object. This is sometimes necessary to facilitate the translation.

    Examples
    --------
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> x = PyccelSymbol('x')
    x
    """
    __slots__ = ('_is_temp',)

    def __new__(cls, name, is_temp=False):
        return super().__new__(cls, name)

    def __init__(self, name, is_temp=False):
        self._is_temp = is_temp
        super().__init__()

    @property
    def is_temp(self):
        """
        Indicates if this symbol represents a temporary variable created by Pyccel,
        and was not present in the original Python code [default value : False].
        """
        return self._is_temp

class PrecomputedCode(PyccelAstNode):
    """
    Internal helper class for storing code which must be defined by the printer
    before it is needed chronologically (e.g. for inline functions as arguments
    to the same function).
    This class should be avoided if at all possible as it may break code which
    searches through attribute nodes, where possible use PyccelAstNode's methods,
    e.g. substitute

    Parameters
    ----------
    code : str
           A string containing the precomputed code
    """
    __slots__ = ('_code',)
    _attribute_nodes = ()

    def __init__(self, code):
        self._code = code
        super().__init__()

    def __str__(self):
        return self._code

    @property
    def code(self):
        """ The string containing the precomputed code
        """
        return self._code


class Iterable(TypedAstNode):
    """
    Wrapper around iterable types helping to convert between those types and a range.

    Wrapper around iterable types helping to convert between those
    types and a range (necessary in low level languages, e.g. C and Fortran).

    If an iterable can be iterated over using a range then this can be done automatically
    by defining the following 3 functions:

    - get_range : Returns the range required for the iteration.
    - get_python_iterable_item : Returns the item of the iterable that will be saved to the
        variables which are the loop targets.
        E.g. for the loop:
        >>> for idx, v in enumerate(var)

        this function should return the range index `idx` (this may be set using
        set_loop_counter) and var[idx].
        These objects are used for type deductions.
    - get_assign_targets : Returns any objects that should be assigned to targets
        E.g. for the loop:
        >>> for idx, v in enumerate(var)

        The object `var[idx]` is returned. The printer is then responsible for
        creating the Assign(v, var[idx])
        E.g. for the loop:
        >>> for r,p in zip(r_var,p_var)

        The objects `r_var[idx]` and `p_var[idx]` are returned. The index is retrieved
        from this class where it was set using set_loop_counter.
        The printer is then responsible for creating the Assign(r, r_var[idx]) and
        Assign(p, p_var[idx]).

    Parameters
    ----------
    num_indices_required : int
        The number of indices that the semantic stage should generate to correctly
        iterate over the object.
    """
    __slots__ = ('_indices', '_num_indices_required')
    _attribute_nodes = ('_indices',)
    _class_type = SymbolicType()
    _shape = None

    def __init__(self, num_indices_required):
        assert isinstance(num_indices_required, int)
        self._indices  = None
        self._num_indices_required = num_indices_required

        super().__init__()

    @property
    def num_loop_counters_required(self):
        """
        Number of indices that should be generate by the semantic stage.

        Number of indices which must be generated in order to convert this
        iterable to a range. This is usually 1.
        """
        return self._num_indices_required

    def set_loop_counter(self, *indices):
        """
        Set the iterator(s) for the generated range.

        These are iterators generated by Pyccel that were not needed for the
        original Python code. Ideally they will also not be necessary in the
        generated Python code so these objects should only be inserted into
        the scope during printing.

        Parameters
        ----------
        *indices : TypedAstNode
            The iterator(s) generated by Pyccel.
        """
        assert self._indices is None
        for i in indices:
            i.set_current_user_node(self)
        self._indices = indices

    @property
    def loop_counters(self):
        """
        Returns the iterator(s) of the generated range.

        Returns the iterator(s) of the generated range.
        """
        return self._indices

    @property
    def modified_args(self):
        """
        Return a tuple of all the arguments which may be modified by this function.

        Return a tuple of all the arguments which may be modified by this function.
        This is notably useful in order to determine the constness of arguments.
        """
        return ()

def symbols(names):
    """
    Transform strings into instances of PyccelSymbol class.

    function returns a sequence of symbols with names taken
    from argument, which can be a comma delimited
    string

    Parameters
    ----------
    name : String
        comma delimited string

    Return
    ----------
    Tuple :
        tuple of instances of PyccelSymbol
    Examples
    --------
    >>> from pyccel.ast.internals import symbols
    >>> x, y, z = symbols('x,y,z')
    (x, y, z)
    """
    names = names.split(',')
    symbols = [PyccelSymbol(name.strip()) for name in names]
    return tuple(symbols)

