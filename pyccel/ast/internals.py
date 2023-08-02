# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
File containing basic classes which are used throughout pyccel.
To avoid circular imports this file should only import from basic, datatypes, and literals
"""

from operator import attrgetter
from pyccel.utilities.stage import PyccelStage

from .basic     import Basic, PyccelAstNode, Immutable
from .datatypes import NativeInteger, default_precision
from .literals  import LiteralInteger

pyccel_stage = PyccelStage()

__all__ = (
    'PrecomputedCode',
    'PyccelArraySize',
    'PyccelArrayShapeElement',
    'PyccelInternalFunction',
    'PyccelSymbol',
    'Slice',
    'get_final_precision',
    'max_precision',
)


class PyccelInternalFunction(PyccelAstNode):
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


class PyccelArraySize(PyccelInternalFunction):
    """
    Gets the total number of elements in an array.

    Class representing a call to a function which would return
    the total number of elements in a multi-dimensional array.

    Parameters
    ----------
    arg : PyccelAstNode
        An array of unknown size.
    """
    __slots__ = ()
    name = 'size'

    _dtype = NativeInteger()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None

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


class PyccelArrayShapeElement(PyccelInternalFunction):
    """
    Gets the number of elements in a given dimension of an array.

    Class representing a call to a function which would return
    the shape of a multi-dimensional array in a given dimension.

    Parameters
    ----------
    arg : PyccelAstNode
        An array of unknown shape.

    index : int
        The dimension along which the shape should be provided.
    """
    __slots__ = ()
    name = 'shape'

    _dtype = NativeInteger()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None

    def __init__(self, arg, index):
        if not isinstance(arg, PyccelAstNode):
            raise TypeError(f'Unknown type {type(arg)} of {arg}.')

        if isinstance(index, int):
            index = LiteralInteger(index)
        elif not isinstance(index, PyccelAstNode):
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

    def __str__(self):
        return f'Shape({self.arg}, {self.index})'

    def __eq__(self, other):
        if isinstance(other, PyccelArrayShapeElement):
            return self.arg == other.arg and self.index == other.index
        else:
            return False


class Slice(Basic):
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
        if start is not None and not (hasattr(start, 'dtype') and isinstance(start.dtype, NativeInteger)):
            raise TypeError('Slice start must be Integer or None')
        if stop is not None and not (hasattr(stop, 'dtype') and isinstance(stop.dtype, NativeInteger)):
            raise TypeError('Slice stop must be Integer or None')
        if step is not None and not (hasattr(step, 'dtype') and isinstance(step.dtype, NativeInteger)):
            raise TypeError('Slice step must be Integer or None')
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
        return f'{start} : {stop}'


class PyccelSymbol(str, Immutable):
    """Symbolic placeholder for a Python variable, which has a name but no type yet.
    This is very generic, and it can also represent a function or a module.

    Parameters
    ----------
    name : String
        name of the symbol

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


class PrecomputedCode(Basic):
    """
    Internal helper class for storing code which must be defined by the printer
    before it is needed chronologically (e.g. for inline functions as arguments
    to the same function).
    This class should be avoided if at all possible as it may break code which
    searches through attribute nodes, where possible use Basic's methods,
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


def max_precision(objs : list, dtype = None, allow_native = True):
    """
    Returns the largest precision of an object in the list

    Parameters
    ----------
    objs : list
           A list of PyccelAstNodes
    dtype : Dtype class
            If this argument is provided then only the
            precision of objects with this dtype are
            considered
    """
    if allow_native and all(o.precision == -1 for o in objs):
        return -1
    elif dtype:
        def_prec = default_precision[str(dtype)]
        return max(def_prec if o.precision == -1 \
                else o.precision for o in objs if o.dtype is dtype)
    else:
        ndarray_list = [o for o in objs if getattr(o, 'is_ndarray', False)]
        if ndarray_list:
            return get_final_precision(max(ndarray_list, key=attrgetter('precision')))
        return max(get_final_precision(o) for o in objs)


def get_final_precision(obj):
    """
    Get the the usable precision of an object. Ie. the precision that you
    can use to print, eg 8 instead of -1 for a default precision float

    If the precision is set to the default then the value of the default
    precision is returned, otherwise the provided precision is returned
    """
    return default_precision[str(obj.dtype)] if obj.precision == -1 else obj.precision
