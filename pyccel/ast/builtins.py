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

from .basic     import Basic, TypedAstNode
from .datatypes import (NativeInteger, NativeBool, NativeFloat,
                        NativeComplex, NativeString, str_dtype,
                        NativeGeneric)
from .internals import PyccelInternalFunction, max_precision, Slice
from .literals  import LiteralInteger, LiteralFloat, LiteralComplex, Nil
from .literals  import Literal, LiteralImaginaryUnit, get_default_literal_value
from .literals  import LiteralString
from .operators import PyccelAdd, PyccelAnd, PyccelMul, PyccelIsNot
from .operators import PyccelMinus, PyccelUnarySub, PyccelNot
from .variable  import IndexedElement

pyccel_stage = PyccelStage()

__all__ = (
    'Lambda',
    'PythonAbs',
    'PythonComplexProperty',
    'PythonReal',
    'PythonImag',
    'PythonConjugate',
    'PythonBool',
    'PythonComplex',
    'PythonEnumerate',
    'PythonFloat',
    'PythonInt',
    'PythonTuple',
    'PythonLen',
    'PythonList',
    'PythonMap',
    'PythonPrint',
    'PythonRange',
    'PythonSum',
    'PythonType',
    'PythonZip',
    'PythonMax',
    'PythonMin',
    'python_builtin_datatype'
)

#==============================================================================
class PythonComplexProperty(PyccelInternalFunction):
    """Represents a call to the .real or .imag property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
    """
    __slots__ = ()
    _dtype = NativeFloat()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def internal_var(self):
        """Return the variable on which the function was called"""
        return self._args[0]

#==============================================================================
class PythonReal(PythonComplexProperty):
    """Represents a call to the .real property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
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
        return 'Real({0})'.format(str(self.internal_var))

#==============================================================================
class PythonImag(PythonComplexProperty):
    """Represents a call to the .imag property

    e.g:
    > a = 1+2j
    > a.imag
    1.0

    arg : Variable, Literal
    """
    __slots__ = ()
    name = 'imag'
    def __new__(cls, arg):
        if arg.dtype is not NativeComplex():
            return get_default_literal_value(arg.dtype)
        else:
            return super().__new__(cls)

    def __str__(self):
        return 'Imag({0})'.format(str(self.internal_var))

#==============================================================================
class PythonConjugate(PyccelInternalFunction):
    """
    Represents a call to the .conjugate() function.

    Represents a call to the conjugate function which is a member of
    the builtin types int, float, complex. The conjugate function is
    called from Python as follows:

    > a = 1+2j
    > a.conjugate()
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
        return 'Conjugate({0})'.format(str(self.internal_var))

#==============================================================================
class PythonBool(TypedAstNode):
    """ Represents a call to Python's native bool() function.
    """
    __slots__ = ('_arg',)
    name = 'bool'
    _dtype = NativeBool()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _attribute_nodes = ('_arg',)

    def __new__(cls, arg):
        if getattr(arg, 'is_optional', None):
            bool_expr = super().__new__(cls)
            bool_expr.__init__(arg)
            return PyccelAnd(PyccelIsNot(arg, Nil()), bool_expr)
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        return self._arg

    def __str__(self):
        return 'Bool({})'.format(str(self.arg))

#==============================================================================
class PythonComplex(TypedAstNode):
    """ Represents a call to Python's native complex() function.
    """
    __slots__ = ('_real_part', '_imag_part', '_internal_var', '_is_cast')
    name = 'complex'

    _dtype = NativeComplex()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
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
        return "complex({}, {})".format(str(self.real), str(self.imag))

#==============================================================================
class PythonEnumerate(Basic):

    """
    Represents the enumerate stmt

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
class PythonFloat(TypedAstNode):
    """ Represents a call to Python's native float() function.
    """
    __slots__ = ('_arg')
    name = 'float'
    _dtype = NativeFloat()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _attribute_nodes = ('_arg',)

    def __new__(cls, arg):
        if isinstance(arg, LiteralFloat) and arg.precision == cls._precision:
            return arg
        if isinstance(arg, (LiteralInteger, LiteralFloat)):
            return LiteralFloat(arg.python_value, precision = cls._precision)
        return super().__new__(cls)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        return self._arg

    def __str__(self):
        return 'float({0})'.format(str(self.arg))

#==============================================================================
class PythonInt(TypedAstNode):
    """ Represents a call to Python's native int() function.
    """

    __slots__ = ('_arg')
    name = 'int'
    _dtype = NativeInteger()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _attribute_nodes  = ('_arg',)

    def __new__(cls, arg):
        if isinstance(arg, LiteralInteger):
            return LiteralInteger(arg.python_value, precision = cls._precision)
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        return self._arg

#==============================================================================
class PythonTuple(TypedAstNode):
    """ Represents a call to Python's native tuple() function.
    """
    __slots__ = ('_args','_inconsistent_shape','_is_homogeneous',
            '_dtype','_precision','_rank','_shape','_order')
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
        arg0 = args[0]
        is_homogeneous = arg0.dtype is not NativeGeneric() and \
                         all(a.dtype is not NativeGeneric() and \
                             arg0.dtype == a.dtype and \
                             arg0.rank  == a.rank  and \
                             arg0.order == a.order for a in args[1:])
        self._inconsistent_shape = not all(arg0.shape==a.shape   for a in args[1:])
        self._is_homogeneous = is_homogeneous
        if is_homogeneous:
            integers  = [a for a in args if a.dtype is NativeInteger()]
            floats    = [a for a in args if a.dtype is NativeFloat()]
            complexes = [a for a in args if a.dtype is NativeComplex()]
            bools     = [a for a in args if a.dtype is NativeBool()]
            strs      = [a for a in args if a.dtype is NativeString()]
            if strs:
                self._dtype = NativeString()
                self._precision = 0
                self._rank  = 0
                self._shape = None
            else:
                if complexes:
                    self._dtype     = NativeComplex()
                    self._precision = max_precision(complexes)
                elif floats:
                    self._dtype     = NativeFloat()
                    self._precision = max_precision(floats)
                elif integers:
                    self._dtype     = NativeInteger()
                    self._precision = max_precision(integers)
                elif bools:
                    self._dtype     = NativeBool()
                    self._precision  = max_precision(bools)
                else:
                    raise TypeError('cannot determine the type of {}'.format(self))


                inner_shape = [() if a.rank == 0 else a.shape for a in args]
                self._rank = max(a.rank for a in args) + 1
                self._shape = (LiteralInteger(len(args)), ) + inner_shape[0]
                self._rank  = len(self._shape)

        else:
            self._rank      = max(a.rank for a in args) + 1
            self._dtype     = NativeGeneric()
            self._precision = 0
            if self._rank == 1:
                self._shape     = (LiteralInteger(len(args)), )
            else:
                self._shape     = (LiteralInteger(len(args)), ) + args[0].shape

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
            raise NotImplementedError("Can't index PythonTuple with type {}".format(type(i)))

    def __add__(self,other):
        return PythonTuple(*(self._args + other._args))

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

    def __str__(self):
        return '({})'.format(', '.join(str(a) for a in self))

    def __repr__(self):
        return 'PythonTuple({})'.format(', '.join(str(a) for a in self))

    @property
    def is_homogeneous(self):
        return self._is_homogeneous

    @property
    def inconsistent_shape(self):
        return self._inconsistent_shape

    @property
    def args(self):
        """ Arguments of the tuple
        """
        return self._args

    @property
    def allows_negative_indexes(self):
        """ Indicates whether variables used to
        index this Variable can be negative
        """
        return False

#==============================================================================
class PythonLen(PyccelInternalFunction):

    """
    Represents a 'len' expression in the code.
    """

    __slots__ = ()
    name      = 'len'
    _dtype     = NativeInteger()
    _precision = -1
    _rank      = 0
    _shape     = None
    _order     = None

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def arg(self):
        return self._args[0]

    def __str__(self):
        return 'len({})'.format(str(self.arg))

#==============================================================================
class PythonList(PythonTuple):
    """ Represents a call to Python's native list() function.
    """
    __slots__ = ()

#==============================================================================
class PythonMap(Basic):
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
class PythonPrint(Basic):

    """Represents a print function in the code.

    expr : TypedAstNode
        The expression to print
    file: String (Optional)
        Select 'stdout' (default) or 'stderr' to print to
    Examples

    >>> from pyccel.ast.internals import symbols
    >>> from pyccel.ast.core import Print
    >>> n,m = symbols('n,m')
    >>> Print(('results', n,m))
    Print((results, n, m))
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
        return self._expr

    @property
    def file(self):
        """ returns the output unit (`stdout` or `stderr`)
        """
        return self._file

#==============================================================================
class PythonRange(Basic):

    """
    Represents a range.

    Examples

    >>> from pyccel.ast.core import Variable
    >>> from pyccel.ast.core import Range
    >>> from pyccel.ast.internals import PyccelSymbol
    >>> s = Variable('int', 's')
    >>> e = PyccelSymbol('e')
    >>> Range(s, e, 1)
    Range(0, n, 1)
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
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def __getitem__(self, index):
        return index


#==============================================================================
class PythonZip(PyccelInternalFunction):

    """
    Represents a zip stmt.

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
    """Represents a call to  python abs for code generation.

    arg : Variable
    """
    __slots__ = ('_dtype','_precision','_rank','_shape','_order')
    name = 'abs'
    def __init__(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = NativeInteger() if x.dtype is NativeInteger() else NativeFloat()
        self._precision = -1
        self._order     = x.order
        super().__init__(x)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonSum(PyccelInternalFunction):
    """Represents a call to  python sum for code generation.

    arg : list , tuple , PythonTuple, List, Variable
    """
    __slots__ = ('_dtype','_precision')
    name   = 'sum'
    _rank  = 0
    _shape = None
    _order = None

    def __init__(self, arg):
        if not isinstance(arg, TypedAstNode):
            raise TypeError('Unknown type of  %s.' % type(arg))
        self._dtype = arg.dtype
        self._precision = -1
        super().__init__(arg)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class PythonMax(PyccelInternalFunction):
    """Represents a call to  python max for code generation.

    arg : list , tuple , PythonTuple, List
    """
    __slots__ = ('_dtype','_precision')
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
            raise TypeError('Unknown type of  %s.' % type(x))
        if not x.is_homogeneous:
            types = ', '.join('{}({})'.format(xi.dtype,xi.precision) for xi in x)
            raise PyccelError("Cannot determine final dtype of 'max' call with arguments of different "
                             "types ({}). Please cast arguments to the desired dtype".format(types))
        self._dtype     = x.dtype
        self._precision = x.precision
        super().__init__(x)


#==============================================================================
class PythonMin(PyccelInternalFunction):
    """Represents a call to  python min for code generation.

    arg : list , tuple , PythonTuple, List, Variable
    """
    __slots__ = ('_dtype','_precision')
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
            raise TypeError('Unknown type of  %s.' % type(x))
        if not x.is_homogeneous:
            types = ', '.join('{}({})'.format(xi.dtype,xi.precision) for xi in x)
            raise PyccelError("Cannot determine final dtype of 'min' call with arguments of different "
                              "types ({}). Please cast arguments to the desired dtype".format(types))
        self._dtype     = x.dtype
        self._precision = x.precision
        super().__init__(x)

#==============================================================================
class Lambda(Basic):
    """Represents a call to python lambda for temporary functions

    Parameters
    ==========
    variables : tuple of symbols
                The arguments to the lambda expression
    expr      : TypedAstNode
                The expression carried out when the lambda function is called
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
        return "{args} -> {expr}".format(args=self.variables,
                expr = self.expr)

#==============================================================================
class PythonType(Basic):
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
            raise PyccelError("Python's type function is not implemented for {} object".format(type(obj)))
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
        dtype = str(self.dtype)
        if prec in (None, -1):
            return LiteralString(f"<class '{dtype}'>")

        precision = prec * (16 if self.dtype is NativeComplex() else 8)
        if self._obj.rank > 0:
            return LiteralString(f"<class 'numpy.ndarray' ({dtype}{precision})>")
        else:
            return LiteralString(f"<class 'numpy.{dtype}{precision}'>")

#==============================================================================
python_builtin_datatypes_dict = {
    'bool'   : PythonBool,
    'float'  : PythonFloat,
    'int'    : PythonInt,
    'complex': PythonComplex
}

def python_builtin_datatype(name):
    """
    Given a symbol name, return the corresponding datatype.

    name: str
        Datatype as written in Python.

    """
    if not isinstance(name, str):
        raise TypeError('name must be a string')

    if name in python_builtin_datatypes_dict:
        return python_builtin_datatypes_dict[name]

    return None

builtin_functions_dict = {
    'abs'      : PythonAbs,
    'range'    : PythonRange,
    'zip'      : PythonZip,
    'enumerate': PythonEnumerate,
    'int'      : PythonInt,
    'float'    : PythonFloat,
    'complex'  : PythonComplex,
    'bool'     : PythonBool,
    'sum'      : PythonSum,
    'len'      : PythonLen,
    'max'      : PythonMax,
    'min'      : PythonMin,
    'not'      : PyccelNot,
    'map'      : PythonMap,
    'type'     : PythonType,
}
