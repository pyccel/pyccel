# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
The Python interpreter has a number of built-in functions and types that are
always available.

In this module we implement some of them in alphabetical order.

"""
from pyccel.errors.errors import PyccelError

from pyccel.utilities.stage import PyccelStage

from .basic     import PyccelAstNode, TypedAstNode
from .datatypes import (NativeInteger, NativeBool, NativeFloat,
                        NativeComplex, NativeGeneric)
from .datatypes import NativeHomogeneousTuple, NativeInhomogeneousTuple
from .datatypes import NativeHomogeneousList, NativeTuple, NativeHomogeneousSet
from .internals import PyccelInternalFunction, Slice, get_final_precision, PyccelArrayShapeElement
from .literals  import LiteralInteger, LiteralFloat, LiteralComplex, Nil
from .literals  import Literal, LiteralImaginaryUnit, convert_to_literal
from .literals  import LiteralString
from .operators import PyccelAdd, PyccelAnd, PyccelMul, PyccelIsNot
from .operators import PyccelMinus, PyccelUnarySub, PyccelNot
from .variable  import IndexedElement, Variable, InhomogeneousTupleVariable

pyccel_stage = PyccelStage()

__all__ = (
    'Lambda',
    'PythonAbs',
    'PythonBool',
    'PythonComplex',
    'PythonComplexProperty',
    'PythonConjugate',
    'PythonEnumerate',
    'PythonFloat',
    'PythonImag',
    'PythonInt',
    'PythonLen',
    'PythonList',
    'PythonMap',
    'PythonMax',
    'PythonMin',
    'PythonPrint',
    'PythonRange',
    'PythonReal',
    'PythonSet',
    'PythonSum',
    'PythonTuple',
    'PythonTupleFunction',
    'PythonType',
    'PythonZip',
    'builtin_functions_dict',
)

#==============================================================================
class PythonComplexProperty(PyccelInternalFunction):
    """
    Represents a call to the .real or .imag property.

    Represents a call to a property of a complex number. The relevant properties
    are the `.real` and `.imag` properties.

    e.g:
    >>> a = 1+2j
    >>> a.real
    1.0

    Parameters
    ----------
    arg : TypedAstNode
        The object which the property is called from.
    """
    __slots__ = ()
    _dtype = NativeFloat()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeFloat()

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def internal_var(self):
        """Return the variable on which the function was called"""
        return self._args[0]

#==============================================================================
class PythonReal(PythonComplexProperty):
    """
    Represents a call to the .real property.

    e.g:
    >>> a = 1+2j
    >>> a.real
    1.0

    Parameters
    ----------
    arg : TypedAstNode
        The object which the property is called from.
    """
    __slots__ = ()
    name = 'real'
    def __new__(cls, arg):
        if isinstance(arg.dtype, NativeBool):
            return PythonInt(arg)
        elif not isinstance(arg.dtype, NativeComplex):
            return arg
        else:
            return super().__new__(cls)

    def __str__(self):
        return f'Real({self.internal_var})'

#==============================================================================
class PythonImag(PythonComplexProperty):
    """
    Represents a call to the .imag property.

    Represents a call to the .imag property of an object with a complex type.
    e.g:
    >>> a = 1+2j
    >>> a.imag
    1.0

    Parameters
    ----------
    arg : TypedAstNode
        The object on which the property is called.
    """
    __slots__ = ()
    name = 'imag'
    def __new__(cls, arg):
        if arg.dtype is not NativeComplex():
            return convert_to_literal(0, dtype = arg.dtype)
        else:
            return super().__new__(cls)

    def __str__(self):
        return f'Imag({self.internal_var})'

#==============================================================================
class PythonConjugate(PyccelInternalFunction):
    """
    Represents a call to the .conjugate() function.

    Represents a call to the conjugate function which is a member of
    the builtin types int, float, complex. The conjugate function is
    called from Python as follows:

    >>> a = 1+2j
    >>> a.conjugate()
    1-2j

    Parameters
    ----------
    arg : TypedAstNode
        The variable/expression which was passed to the
        conjugate function.
    """
    __slots__ = ()
    _dtype = NativeComplex()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeComplex()
    name = 'conjugate'

    def __new__(cls, arg):
        if arg.dtype is NativeBool():
            return PythonInt(arg)
        elif arg.dtype is not NativeComplex():
            return arg
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def internal_var(self):
        """Return the variable on which the function was called"""
        return self._args[0]

    def __str__(self):
        return f'Conjugate({self.internal_var})'

#==============================================================================
class PythonBool(PyccelInternalFunction):
    """
    Represents a call to Python's native `bool()` function.

    Represents a call to Python's native `bool()` function which casts an
    argument to a boolean.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    name = 'bool'
    _dtype = NativeBool()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeBool()

    def __new__(cls, arg):
        if getattr(arg, 'is_optional', None):
            bool_expr = super().__new__(cls)
            bool_expr.__init__(arg)
            return PyccelAnd(PyccelIsNot(arg, Nil()), bool_expr)
        else:
            return super().__new__(cls)

    @property
    def arg(self):
        """
        Get the argument which was passed to the function.

        Get the argument which was passed to the function.
        """
        return self._args[0]

    def __str__(self):
        return f'Bool({self.arg})'

#==============================================================================
class PythonComplex(PyccelInternalFunction):
    """
    Represents a call to Python's native `complex()` function.

    Represents a call to Python's native `complex()` function which casts an
    argument to a complex number.

    Parameters
    ----------
    arg0 : TypedAstNode
        The first argument passed to the function (either a real or a complex).

    arg1 : TypedAstNode, default=0
        The second argument passed to the function (the imaginary part).
    """
    __slots__ = ('_real_part', '_imag_part', '_internal_var', '_is_cast')
    name = 'complex'

    _dtype = NativeComplex()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeComplex()
    _real_cast = PythonReal
    _imag_cast = PythonImag
    _attribute_nodes = ('_real_part', '_imag_part', '_internal_var')

    def __new__(cls, arg0, arg1=LiteralFloat(0)):

        if isinstance(arg0, Literal) and isinstance(arg1, Literal):
            real_part = 0
            imag_part = 0

            # Collect real and imag part from first argument
            if isinstance(arg0, LiteralComplex):
                real_part += arg0.real.python_value
                imag_part += arg0.imag.python_value
            else:
                real_part += arg0.python_value

            # Collect real and imag part from second argument
            if isinstance(arg1, LiteralComplex):
                real_part -= arg1.imag.python_value
                imag_part += arg1.real.python_value
            else:
                imag_part += arg1.python_value

            return LiteralComplex(real_part, imag_part, precision = cls._precision)


        # Split arguments depending on their type to ensure that the arguments are
        # either a complex and LiteralFloat(0) or 2 floats

        if arg0.dtype is NativeComplex() and arg1.dtype is NativeComplex():
            # both args are complex
            return PyccelAdd(arg0, PyccelMul(arg1, LiteralImaginaryUnit()))
        return super().__new__(cls)

    def __init__(self, arg0, arg1 = LiteralFloat(0)):
        self._is_cast = arg0.dtype is NativeComplex() and \
                        isinstance(arg1, Literal) and arg1.python_value == 0

        if self._is_cast:
            self._real_part = self._real_cast(arg0)
            self._imag_part = self._imag_cast(arg0)
            self._internal_var = arg0

        else:
            self._internal_var = None

            if arg0.dtype is NativeComplex() and \
                    not (isinstance(arg1, Literal) and arg1.python_value == 0):
                # first arg is complex. Second arg is non-0
                self._real_part = self._real_cast(arg0)
                self._imag_part = PyccelAdd(self._imag_cast(arg0), arg1)
            elif arg1.dtype is NativeComplex():
                if isinstance(arg0, Literal) and arg0.python_value == 0:
                    # second arg is complex. First arg is 0
                    self._real_part = PyccelUnarySub(self._imag_cast(arg1))
                    self._imag_part = self._real_cast(arg1)
                else:
                    # Second arg is complex. First arg is non-0
                    self._real_part = PyccelMinus(arg0, self._imag_cast(arg1))
                    self._imag_part = self._real_cast(arg1)
            else:
                self._real_part = self._real_cast(arg0)
                self._imag_part = self._real_cast(arg1)
        super().__init__()

    @property
    def is_cast(self):
        """ Indicates if the function is casting or assembling a complex """
        return self._is_cast

    @property
    def real(self):
        """ Returns the real part of the complex """
        return self._real_part

    @property
    def imag(self):
        """ Returns the imaginary part of the complex """
        return self._imag_part

    @property
    def internal_var(self):
        """ When the complex call is a cast, returns the variable being cast """
        assert(self._is_cast)
        return self._internal_var

    def __str__(self):
        return f"complex({self.real}, {self.imag})"

#==============================================================================
class PythonEnumerate(PyccelAstNode):
    """
    Represents a call to Python's native `enumerate()` function.

    Represents a call to Python's native `enumerate()` function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.

    start : TypedAstNode
        The start value of the enumeration index.
    """
    __slots__ = ('_element','_start')
    _attribute_nodes = ('_element','_start')
    name = 'enumerate'

    def __init__(self, arg, start = None):
        if pyccel_stage != "syntactic" and \
                not isinstance(arg, TypedAstNode):
            raise TypeError('Expecting an arg of valid type')
        self._element = arg
        self._start   = start or LiteralInteger(0)
        super().__init__()

    @property
    def element(self):
        """
        Get the object which is being enumerated.

        Get the object which is being enumerated.
        """
        return self._element

    @property
    def start(self):
        """ Returns the value from which the indexing starts
        """
        return self._start

    def __getitem__(self, index):
        return [PyccelAdd(index, self.start, simplify=True),
                self.element[index]]

    @property
    def length(self):
        """ Return the length of the enumerated object
        """
        return PythonLen(self.element)

#==============================================================================
class PythonFloat(PyccelInternalFunction):
    """
    Represents a call to Python's native `float()` function.

    Represents a call to Python's native `float()` function which casts an
    argument to a floating point number.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    name = 'float'
    _dtype = NativeFloat()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeFloat()

    def __new__(cls, arg):
        if isinstance(arg, LiteralFloat) and arg.precision == cls._precision:
            return arg
        if isinstance(arg, (LiteralInteger, LiteralFloat)):
            return LiteralFloat(arg.python_value, precision = cls._precision)
        return super().__new__(cls)

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def arg(self):
        """
        Get the argument which was passed to the function.

        Get the argument which was passed to the function.
        """
        return self._args[0]

    def __str__(self):
        return f'float({self.arg})'

#==============================================================================
class PythonInt(PyccelInternalFunction):
    """
    Represents a call to Python's native `int()` function.

    Represents a call to Python's native `int()` function which casts an
    argument to an integer.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """

    __slots__ = ()
    name = 'int'
    _dtype = NativeInteger()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeInteger()

    def __new__(cls, arg):
        if isinstance(arg, LiteralInteger):
            return LiteralInteger(arg.python_value, precision = cls._precision)
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def arg(self):
        """
        Get the argument which was passed to the function.

        Get the argument which was passed to the function.
        """
        return self._args[0]

#==============================================================================
class PythonTuple(TypedAstNode):
    """
    Class representing a call to Python's native (,) function which creates tuples.

    Class representing a call to Python's native (,) function
    which initialises a literal tuple.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the tuple function.
    """
    __slots__ = ('_args','_is_homogeneous',
            '_dtype','_precision','_rank','_shape','_order', '_class_type')
    _iterable        = True
    _attribute_nodes = ('_args',)

    def __init__(self, *args):
        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        elif len(args) == 0:
            self._dtype = NativeGeneric()
            self._precision = 0
            self._rank  = 0
            self._shape = None
            self._order = None
            self._is_homogeneous = False
            return

        dtypes = set(a.dtype for a in args)
        precisions = set(get_final_precision(a) for a in args)
        class_types = set(a.class_type for a in args)
        ranks = set(a.rank for a in args)
        orders = set(a.order for a in args)
        shapes = set(tuple(si if not (isinstance(si, PyccelArrayShapeElement) or \
                           si.get_attribute_nodes(PyccelArrayShapeElement)) \
                        else None for si in a.shape) \
                     if a.shape is not None else None for a in args)
        is_homogeneous = len(dtypes) == 1 and len(precisions) == 1 and \
                         len(class_types) == 1 and len(ranks) == 1 and \
                         len(orders) == 1 and len(shapes) == 1 and \
                         NativeGeneric() not in dtypes
        contains_pointers = any(isinstance(a, (Variable, IndexedElement)) and a.rank>0 and \
                            not isinstance(a.dtype, NativeHomogeneousTuple) for a in args)

        self._is_homogeneous = is_homogeneous
        if is_homogeneous and not contains_pointers:
            arg0 = args[0]
            self._dtype = arg0.dtype
            self._precision = arg0.precision
            inner_shape = [() if a.rank == 0 else a.shape for a in args]
            self._rank = max(a.rank for a in args) + 1
            self._shape = (LiteralInteger(len(args)), ) + inner_shape[0]
            self._rank  = len(self._shape)

            self._class_type = NativeHomogeneousTuple()

        else:
            self._rank       = 1
            self._dtype      = NativeInhomogeneousTuple(*[a.dtype for a in args])
            self._precision  = 0
            self._class_type = self._dtype
            self._shape     = (LiteralInteger(len(args)), )

        self._order = None if self._rank < 2 else 'C'

    def __getitem__(self,i):
        def is_int(a):
            return isinstance(a, (int, LiteralInteger)) or \
                    (isinstance(a, PyccelUnarySub) and \
                     isinstance(a.args[0], (int, LiteralInteger)))

        def to_int(a):
            if a is None:
                return None
            elif isinstance(a, PyccelUnarySub):
                return -a.args[0].python_value
            else:
                return a

        if is_int(i):
            return self._args[to_int(i)]
        elif isinstance(i, Slice) and \
                all(is_int(s) or s is None for s in (i.start, i.step, i.stop)):
            return PythonTuple(*self._args[to_int(i.start):to_int(i.stop):to_int(i.step)])
        elif self.is_homogeneous:
            return IndexedElement(self, i)
        else:
            raise NotImplementedError(f"Can't index PythonTuple with type {type(i)}")

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

    def __str__(self):
        args = ', '.join(str(a) for a in self)
        return f'({args})'

    def __repr__(self):
        args = ', '.join(str(a) for a in self)
        return f'PythonTuple({args})'

    @property
    def is_homogeneous(self):
        """
        Indicates whether the tuple is homogeneous or inhomogeneous.

        Indicates whether all elements of the tuple have the same dtype, precision,
        rank, etc (homogenous) or if these values can vary (inhomogeneous).
        """
        return self._is_homogeneous

    @property
    def args(self):
        """
        Arguments of the tuple.

        The arguments that were used to initialise the tuple.
        """
        return self._args

class PythonTupleFunction(TypedAstNode):
    """
    Class representing a call to the `tuple` function.

    Class representing a call to the `tuple` function. This is
    different to the `(,)` syntax as it only takes one argument
    and unpacks any variables.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function call.
    """
    __slots__ = ()
    _attribute_nodes = ()

    def __new__(cls, arg):
        if isinstance(arg, PythonTuple):
            return arg
        elif isinstance(arg, (PythonList, InhomogeneousTupleVariable)):
            return PythonTuple(*arg)
        elif isinstance(arg.shape[0], LiteralInteger):
            return PythonTuple(*[arg[i] for i in range(arg.shape[0])])
        else:
            raise TypeError(f"Can't unpack {arg} into a tuple")

#==============================================================================
class PythonLen(PyccelInternalFunction):
    """
    Represents a `len` expression in the code.

    Represents a call to the function `len` which calculates the length
    (aka the first element of the shape) of an object. This can usually
    be calculated in the generated code, but in an inhomogeneous object
    the integer value of the shape must be returned.

    Parameters
    ----------
    arg : TypedAstNode
        The argument whose length is being examined.
    """
    __slots__ = ()
    def __new__(cls, arg):
        return arg.shape[0]

#==============================================================================
class PythonList(TypedAstNode):
    """
    Class representing a call to Python's `[,]` function.

    Class representing a call to Python's `[,]` function which generates
    a literal Python list.

    Parameters
    ----------
    *args : tuple of TypedAstNodes
        The arguments passed to the operator.

    See Also
    --------
    FunctionalFor
        The `[]` function when it describes a comprehension.
    """
    __slots__ = ('_args','_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_args',)
    _class_type = NativeHomogeneousList()

    def __init__(self, *args):
        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        elif len(args) == 0:
            self._dtype = NativeGeneric()
            self._precision = 0
            self._rank  = 0
            self._shape = None
            self._order = None
            return
        arg0 = args[0]
        precision = get_final_precision(arg0)
        is_homogeneous = arg0.dtype is not NativeGeneric() and \
                         all(a.dtype is not NativeGeneric() and \
                             arg0.dtype == a.dtype and \
                             precision == get_final_precision(a) and \
                             arg0.rank  == a.rank  and \
                             arg0.order == a.order for a in args[1:])
        if is_homogeneous:
            self._dtype = arg0.dtype
            self._precision = arg0.precision

            inner_shape = [() if a.rank == 0 else a.shape for a in args]
            self._rank = max(a.rank for a in args) + 1
            self._shape = (LiteralInteger(len(args)), ) + inner_shape[0]
            self._rank  = len(self._shape)

        else:
            raise TypeError("Can't create an inhomogeneous list")

        self._order = None if self._rank < 2 else 'C'

    def __iter__(self):
        return self._args.__iter__()

    def __str__(self):
        args = ', '.join(str(a) for a in self)
        return f'({args})'

    def __repr__(self):
        args = ', '.join(str(a) for a in self)
        return f'PythonList({args})'

    @property
    def args(self):
        """
        Arguments of the list.

        The arguments that were used to initialise the list.
        """
        return self._args

    @property
    def is_homogeneous(self):
        """
        Indicates whether the list is homogeneous or inhomogeneous.

        Indicates whether all elements of the list have the same dtype, precision,
        rank, etc (homogenous) or if these values can vary (inhomogeneous). Lists
        are always homogeneous.
        """
        return True

#==============================================================================
class PythonSet(TypedAstNode):
    """
    Class representing a call to Python's `{,}` function.

    Class representing a call to Python's `{,}` function which generates
    a literal Python Set.

    Parameters
    ----------
    *args : tuple of TypedAstNodes
        The arguments passed to the operator.
    """
    __slots__ = ('_args','_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_args',)
    _class_type = NativeHomogeneousSet()

    def __init__(self, *args):
        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        arg0 = args[0]
        precision = get_final_precision(arg0)
        is_homogeneous = arg0.dtype is not NativeGeneric() and \
                         all(a.dtype is not NativeGeneric() and \
                             arg0.dtype == a.dtype and \
                             precision == get_final_precision(a) and \
                             arg0.rank  == a.rank  and \
                             arg0.order == a.order for a in args[1:])
        if is_homogeneous:
            self._dtype = arg0.dtype
            self._precision = arg0.precision
            inner_shape = [() if a.rank == 0 else a.shape for a in args]
            self._shape = (LiteralInteger(len(args)), ) + inner_shape[0]
            self._rank  = len(self._shape)
            if self._rank > 1:
                raise TypeError("Pyccel can't hash non-scalar types")
        else:
            raise TypeError("Can't create an inhomogeneous set")

        self._order = None if self._rank < 2 else 'C'

    def __iter__(self):
        return self._args.__iter__()

    @property
    def args(self):
        """
        Arguments of the set.

        The arguments that were used to initialise the set.
        """
        return self._args

    @property
    def is_homogeneous(self):
        """
        Indicates whether the set is homogeneous or inhomogeneous.

        Indicates whether all elements of the Set have the same dtype, precision,
        rank, etc (homogenous) or if these values can vary (inhomogeneous). sets
        are always homogeneous.
        """
        return True

#==============================================================================
class PythonMap(PyccelAstNode):
    """ Represents the map stmt
    """
    __slots__ = ('_func','_func_args')
    _attribute_nodes = ('_func','_func_args')
    name = 'map'

    def __init__(self, func, func_args):
        self._func = func
        self._func_args = func_args
        super().__init__()

    @property
    def func(self):
        """ Arguments of the map
        """
        return self._func

    @property
    def func_args(self):
        """ Arguments of the function
        """
        return self._func_args

    def __getitem__(self, index):
        return self.func, IndexedElement(self.func_args, index)

    @property
    def length(self):
        """ Return the length of the resulting object
        """
        return PythonLen(self.func_args)

#==============================================================================
class PythonPrint(PyccelAstNode):
    """
    Represents a call to the print function in the code.

    Represents a call to the built-in Python function `print` in the code.

    Parameters
    ----------
    expr : TypedAstNode
        The expression to print.
    file : str, default='stdout'
        One of [stdout,stderr].
    """
    __slots__ = ('_expr', '_file')
    _attribute_nodes = ('_expr',)
    name = 'print'

    def __init__(self, expr, file="stdout"):
        if file not in ('stdout', 'stderr'):
            raise ValueError('output_unit can be `stdout` or `stderr`')
        self._expr = expr
        self._file = file
        super().__init__()

    @property
    def expr(self):
        """
        The expression that should be printed.

        The expression that should be printed.
        """
        return self._expr

    @property
    def file(self):
        """ returns the output unit (`stdout` or `stderr`)
        """
        return self._file

#==============================================================================
class PythonRange(PyccelAstNode):
    """
    Class representing a range.

    Class representing a call to the built-in Python function `range`. This function
    is parametrised by an interval (described by a start element and a stop element)
    and a step. The step describes the number of elements between subsequent elements
    in the range.

    Parameters
    ----------
    *args : tuple of TypedAstNodes
        The arguments passed to the range.
        If one argument is passed then it represents the end of the interval.
        If two arguments are passed then they represent the start and end of the interval.
        If three arguments are passed then they represent the start, end and step of the interval.
    """
    __slots__ = ('_start','_stop','_step')
    _attribute_nodes = ('_start', '_stop', '_step')
    name = 'range'

    def __init__(self, *args):
        # Define default values
        n = len(args)

        if n == 1:
            self._start = LiteralInteger(0)
            self._stop  = args[0]
            self._step  = LiteralInteger(1)
        elif n == 2:
            self._start = args[0]
            self._stop  = args[1]
            self._step  = LiteralInteger(1)
        elif n == 3:
            self._start = args[0]
            self._stop  = args[1]
            self._step  = args[2]
        else:
            raise ValueError('Range has at most 3 arguments')

        super().__init__()

    @property
    def start(self):
        """
        Get the start of the interval.

        Get the start of the interval which the range iterates over.
        """
        return self._start

    @property
    def stop(self):
        """
        Get the end of the interval.

        Get the end of the interval which the range iterates over. The
        interval does not include this value.
        """
        return self._stop

    @property
    def step(self):
        """
        Get the step between subsequent elements in the range.

        Get the step between subsequent elements in the range.
        """
        return self._step

    def __getitem__(self, index):
        return index


#==============================================================================
class PythonZip(PyccelInternalFunction):
    """
    Represents a call to Python `zip` for code generation.

    Represents a call to Python's built-in function `zip`.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ('_length',)
    name = 'zip'

    def __init__(self, *args):
        if not isinstance(args, (tuple, list)):
            raise TypeError('args must be a list or tuple')
        elif len(args) < 2:
            raise ValueError('args must be of length > 2')
        super().__init__(*args)
        if pyccel_stage == 'syntactic':
            self._length = None
            return
        else:
            lengths = [a.shape[0].python_value for a in self.args if isinstance(a.shape[0], LiteralInteger)]
            if lengths:
                self._length = min(lengths)
            else:
                self._length = self.args[0].shape[0]

    @property
    def length(self):
        """ Length of the shortest zip argument
        """
        return self._length

    def __getitem__(self, index):
        return [a[index] for a in self.args]

#==============================================================================
class PythonAbs(PyccelInternalFunction):
    """
    Represents a call to Python `abs` for code generation.

    Represents a call to Python's built-in function `abs`.

    Parameters
    ----------
    x : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_dtype','_precision','_rank','_shape','_order','_class_type')
    name = 'abs'
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeFloat()
        self._precision = -1
        self._order     = x.order
        self._class_type = x.class_type
        super().__init__(x)

    @property
    def arg(self):
        """
        The argument passed to the abs function.

        The argument passed to the abs function.
        """
        return self._args[0]

#==============================================================================
class PythonSum(PyccelInternalFunction):
    """
    Represents a call to Python `sum` for code generation.

    Represents a call to Python's built-in function `sum`.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_dtype','_precision','_class_type')
    name   = 'sum'
    _rank  = 0
    _shape = None
    _order = None

    def __init__(self, arg):
        if not isinstance(arg, TypedAstNode):
            raise TypeError(f'Unknown type of {type(arg)}.' )
        self._dtype = arg.dtype
        self._precision = -1
        if isinstance(arg.class_type, (NativeHomogeneousList, NativeTuple)):
            self._class_type = arg.dtype
        else:
            self._class_type = arg.class_type
        super().__init__(arg)

    @property
    def arg(self):
        """
        The argument passed to the sum function.

        The argument passed to the sum function.
        """
        return self._args[0]

#==============================================================================
class PythonMax(PyccelInternalFunction):
    """
    Represents a call to Python's built-in `max` function.

    Represents a call to Python's built-in `max` function.

    Parameters
    ----------
    *x : list, tuple, PythonTuple, PythonList
        The arguments passed to the funciton.
    """
    __slots__ = ('_dtype','_precision','_class_type')
    name   = 'max'
    _rank  = 0
    _shape = None
    _order = None

    def __init__(self, *x):
        if len(x)==1:
            x = x[0]

        if isinstance(x, (list, tuple)):
            x = PythonTuple(*x)
        elif not isinstance(x, (PythonTuple, PythonList)):
            raise TypeError(f'Unknown type of {type(x)}.' )

        if not x.is_homogeneous:
            types = ', '.join('{xi.dtype}({xi.precision})' for xi in x)
            raise PyccelError("Cannot determine final dtype of 'max' call with arguments of different "
                             f"types ({types}). Please cast arguments to the desired dtype")
        self._dtype     = x.dtype
        self._precision = x.precision
        if isinstance(x.class_type, (NativeHomogeneousList, NativeTuple)):
            self._class_type = x.dtype
        else:
            self._class_type = x.class_type
        super().__init__(x)


#==============================================================================
class PythonMin(PyccelInternalFunction):
    """
    Represents a call to Python's built-in `max` function.

    Represents a call to Python's built-in `max` function.

    Parameters
    ----------
    *x : list, tuple, PythonTuple, PythonList
        The arguments passed to the funciton.
    """
    __slots__ = ('_dtype','_precision','_class_type')
    name   = 'min'
    _rank  = 0
    _shape = None
    _order = None
    def __init__(self, *x):
        if len(x)==1:
            x = x[0]

        if isinstance(x, (list, tuple)):
            x = PythonTuple(*x)
        elif not isinstance(x, (PythonTuple, PythonList)):
            raise TypeError(f'Unknown type of {type(x)}.' )

        if not x.is_homogeneous:
            types = ', '.join(f'{xi.dtype}({xi.precision})' for xi in x)
            raise PyccelError("Cannot determine final dtype of 'min' call with arguments of different "
                              f"types ({types}). Please cast arguments to the desired dtype")
        self._dtype     = x.dtype
        self._precision = x.precision
        if isinstance(x.class_type, (NativeHomogeneousList, NativeTuple)):
            self._class_type = x.dtype
        else:
            self._class_type = x.class_type
        super().__init__(x)

#==============================================================================
class Lambda(PyccelAstNode):
    """
    Represents a call to Python's lambda for temporary functions.

    Represents a call to Python's built-in function `lambda` for temporary functions.

    Parameters
    ----------
    variables : tuple of symbols
        The arguments to the lambda expression.
    expr : TypedAstNode
        The expression carried out when the lambda function is called.
    """
    __slots__ = ('_variables', '_expr')
    _attribute_nodes = ('_variables', '_expr')
    def __init__(self, variables, expr):
        if not isinstance(variables, (list, tuple)):
            raise TypeError("Lambda arguments must be a tuple or list")
        self._variables = tuple(variables)
        self._expr = expr
        super().__init__()

    @property
    def variables(self):
        """ The arguments to the lambda function
        """
        return self._variables

    @property
    def expr(self):
        """ The expression carried out when the lambda function is called
        """
        return self._expr

    def __call__(self, *args):
        """ Returns the expression with the arguments replaced with
        the calling arguments
        """
        assert(len(args) == len(self.variables))
        return self.expr.subs(self.variables, args)

    def __str__(self):
        return f"{self.variables} -> {self.expr}"

#==============================================================================
class PythonType(PyccelAstNode):
    """
    Represents a call to the Python builtin `type` function.

    The use of `type` in code is usually for one of two purposes.
    Firstly it is useful for debugging. In this case the `print_string`
    property is useful to obtain the underlying type. It is
    equally useful to provide datatypes to objects in templated
    functions. This double usage should be considered when using
    this class.

    Parameters
    ==========
    obj : TypedAstNode
          The object whose type we wish to investigate.
    """
    __slots__ = ('_dtype','_precision','_obj')
    _attribute_nodes = ('_obj',)

    def __init__(self, obj):
        if not isinstance (obj, TypedAstNode):
            raise PyccelError(f"Python's type function is not implemented for {type(obj)} object")
        self._dtype = obj.dtype
        self._precision = obj.precision
        self._obj = obj

        super().__init__()

    @property
    def dtype(self):
        """ Returns the dtype of this type
        """
        return self._dtype

    @property
    def precision(self):
        """ Returns the precision of this type
        """
        return self._precision

    @property
    def arg(self):
        """ Returns the object for which the type is determined
        """
        return self._obj

    @property
    def print_string(self):
        """
        Return a LiteralString describing the type.

        Constructs a LiteralString containing the message usually
        printed by Python to describe this type. This string can
        then be easily printed in each language.
        """
        prec = self.precision
        dtype = self.dtype.name
        if prec in (None, 0, -1):
            return LiteralString(f"<class '{dtype}'>")

        class_type = self._obj.class_type

        precision = prec * (16 if self.dtype is NativeComplex() else 8)
        if class_type != self.dtype:
            return LiteralString(f"<class '{class_type}' ({dtype}{precision})>")
        else:
            return LiteralString(f"<class 'numpy.{dtype}{precision}'>")

#==============================================================================

builtin_functions_dict = {
    'abs'      : PythonAbs,
    'bool'     : PythonBool,
    'range'    : PythonRange,
    'zip'      : PythonZip,
    'enumerate': PythonEnumerate,
    'int'      : PythonInt,
    'float'    : PythonFloat,
    'complex'  : PythonComplex,
    'bool'     : PythonBool,
    'sum'      : PythonSum,
    'len'      : PythonLen,
    'list'     : PythonList,
    'max'      : PythonMax,
    'min'      : PythonMin,
    'not'      : PyccelNot,
    'map'      : PythonMap,
    'str'      : LiteralString,
    'type'     : PythonType,
    'tuple'    : PythonTupleFunction,
}
