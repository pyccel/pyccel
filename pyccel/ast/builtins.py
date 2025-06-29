# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
The Python interpreter has a number of built-in functions and types that are
always available.

In this module we implement some of them in alphabetical order.

"""
from pyccel.errors.errors import PyccelError

from pyccel.utilities.stage import PyccelStage

from .basic     import PyccelAstNode, TypedAstNode
from .datatypes import PythonNativeInt, PythonNativeBool, PythonNativeFloat
from .datatypes import GenericType, PythonNativeComplex, CharType
from .datatypes import PrimitiveBooleanType, PrimitiveComplexType
from .datatypes import HomogeneousTupleType, InhomogeneousTupleType, TupleType
from .datatypes import HomogeneousListType, HomogeneousContainerType
from .datatypes import FixedSizeNumericType, HomogeneousSetType, SymbolicType
from .datatypes import DictType, VoidType, TypeAlias, StringType
from .datatypes import original_type_to_pyccel_type
from .internals import PyccelFunction, Slice, PyccelArrayShapeElement, Iterable
from .literals  import LiteralInteger, LiteralFloat, LiteralComplex, Nil
from .literals  import Literal, LiteralImaginaryUnit, convert_to_literal
from .literals  import LiteralString
from .operators import PyccelAdd, PyccelAnd, PyccelMul, PyccelIsNot
from .operators import PyccelMinus, PyccelUnarySub, PyccelNot
from .variable  import IndexedElement, Variable

pyccel_stage = PyccelStage()

__all__ = (
    'Lambda',
    'PythonAbs',
    'PythonBool',
    'PythonComplex',
    'PythonComplexProperty',
    'PythonConjugate',
    'PythonDict',
    'PythonDictFunction',
    'PythonEnumerate',
    'PythonFloat',
    'PythonImag',
    'PythonInt',
    'PythonIsInstance',
    'PythonLen',
    'PythonList',
    'PythonListFunction',
    'PythonMap',
    'PythonMax',
    'PythonMin',
    'PythonPrint',
    'PythonRange',
    'PythonReal',
    'PythonRound',
    'PythonSet',
    'PythonSetFunction',
    'PythonStr',
    'PythonSum',
    'PythonTuple',
    'PythonTupleFunction',
    'PythonType',
    'PythonZip',
    'VariableIterator',
    'builtin_functions_dict',
)

#==============================================================================
class PythonComplexProperty(PyccelFunction):
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
    _shape = None
    _class_type = PythonNativeFloat()

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
        if isinstance(arg.dtype, PythonNativeBool):
            return PythonInt(arg)
        elif not isinstance(arg.dtype.primitive_type, PrimitiveComplexType):
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
        if not isinstance(arg.dtype.primitive_type, PrimitiveComplexType):
            return convert_to_literal(0, dtype = arg.dtype)
        else:
            return super().__new__(cls)

    def __str__(self):
        return f'Imag({self.internal_var})'

#==============================================================================
class PythonConjugate(PyccelFunction):
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
    _shape = None
    _class_type = PythonNativeComplex()
    name = 'conjugate'

    def __new__(cls, arg):
        if arg.dtype is PythonNativeBool():
            return PythonInt(arg)
        elif not isinstance(arg.dtype.primitive_type, PrimitiveComplexType):
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
class PythonBool(PyccelFunction):
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
    _static_type = PythonNativeBool()
    _shape = None
    _class_type = PythonNativeBool()

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
class PythonComplex(PyccelFunction):
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

    _static_type = PythonNativeComplex()
    _shape = None
    _class_type = PythonNativeComplex()
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

            return LiteralComplex(real_part, imag_part, dtype = cls._static_type)


        # Split arguments depending on their type to ensure that the arguments are
        # either a complex and LiteralFloat(0) or 2 floats

        if isinstance(arg0.dtype.primitive_type, PrimitiveComplexType) and isinstance(arg1.dtype.primitive_type, PrimitiveComplexType):
            # both args are complex
            return PyccelAdd(arg0, PyccelMul(arg1, LiteralImaginaryUnit()))
        return super().__new__(cls)

    def __init__(self, arg0, arg1 = LiteralFloat(0)):
        self._is_cast = isinstance(arg1, Literal) and arg1.python_value == 0

        if self._is_cast:
            self._real_part = self._real_cast(arg0)
            self._imag_part = self._imag_cast(arg0)
            self._internal_var = arg0

        else:
            self._internal_var = None

            if isinstance(arg0.dtype.primitive_type, PrimitiveComplexType) and \
                    not (isinstance(arg1, Literal) and arg1.python_value == 0):
                # first arg is complex. Second arg is non-0
                self._real_part = self._real_cast(arg0)
                self._imag_part = PyccelAdd(self._imag_cast(arg0), arg1)
            elif isinstance(arg1.dtype.primitive_type, PrimitiveComplexType):
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
        """
        When the complex call is a cast, returns the variable being cast.

        When the complex call is a cast, returns the variable being cast.
        This property should only be used when handling a cast.
        """
        assert self._is_cast
        return self._internal_var

    def __str__(self):
        return f"complex({self.real}, {self.imag})"

#==============================================================================
class PythonEnumerate(Iterable):
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
    _attribute_nodes = Iterable._attribute_nodes + ('_element','_start')
    name = 'enumerate'
    _class_type = SymbolicType()
    _shape = ()

    def __init__(self, arg, start = None):
        if pyccel_stage != "syntactic" and \
                not isinstance(arg, TypedAstNode):
            raise TypeError('Expecting an arg of valid type')
        self._element = arg
        self._start   = start or LiteralInteger(0)
        super().__init__(int(self.start != 0))

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

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns two objects that could be elements of the enumerate.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        index = self._indices[0]
        return [index, self.element[index]]

    def get_assign_targets(self):
        """
        Get objects that should be assigned to variables to use the enumerate targets.

        Get objects that should be assigned to variables to use the enumerate targets.
        If the start of the enumerate is 0 then the only object that needs to be assigned
        is the element of the variable, however if the indexing is offset compared to
        the variable then both the index and the variable need to be created.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        index = self._indices[0]
        if index.is_temp:
            return [PyccelAdd(index, self.start, simplify=True),
                    self.element[index]]
        else:
            return [self.element[index]]

    def get_range(self):
        """
        Get a range that can be used to iterate over the enumerate iterable.

        Get a range that can be used to iterate over the enumerate iterable.

        Returns
        -------
        PythonRange
            A range that can be used to iterate over the enumerate iterable.
        """
        return PythonRange(PythonLen(self.element))

#==============================================================================
class PythonFloat(PyccelFunction):
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
    _static_type = PythonNativeFloat()
    _shape = None
    _class_type = PythonNativeFloat()

    def __new__(cls, arg):
        if isinstance(arg, LiteralFloat) and arg.dtype is cls._static_type:
            return arg
        if isinstance(arg, (LiteralInteger, LiteralFloat)):
            return LiteralFloat(arg.python_value, dtype = cls._static_type)
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

# ===========================================================================
class PythonRound(PyccelFunction):
    """
    Class representing a call to Python's native round() function.

    Class representing a call to Python's native round() function
    which rounds a float or integer to a given number of decimals.

    Parameters
    ----------
    number : TypedAstNode
        The number to be rounded.
    ndigits : TypedAstNode, optional
        The number of digits to round to.
    """
    __slots__ = ('_class_type',)
    name = 'round'
    _rank = 0
    _shape = None
    _order = None

    def __init__(self, number, ndigits = None):
        if ndigits is None or number.class_type.primitive_type is PrimitiveBooleanType():
            self._class_type = PythonNativeInt()
        else:
            self._class_type = number.class_type
        super().__init__(number, ndigits)

    @property
    def arg(self):
        """
        The number to be rounded.

        The number to be rounded.
        """
        return self._args[0]

    @property
    def ndigits(self):
        """
        The number of digits to which the argument is rounded.

        The number of digits to which the argument is rounded.
        """
        return self._args[1]

#==============================================================================
class PythonInt(PyccelFunction):
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
    _static_type = PythonNativeInt()
    _shape = None
    _class_type = PythonNativeInt()

    def __new__(cls, arg):
        if isinstance(arg, LiteralInteger):
            return LiteralInteger(arg.python_value, dtype = cls._static_type)
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
    prefer_inhomogeneous : bool, default=False
        A boolean that can be used to ensure that the tuple is stocked as an
        inhomogeneous object even if it could be homogeneous.
    """
    __slots__ = ('_args','_is_homogeneous', '_shape', '_class_type')
    _iterable = True
    _attribute_nodes = ('_args',)

    def __init__(self, *args, prefer_inhomogeneous = False):
        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        elif len(args) == 0:
            self._class_type = HomogeneousTupleType(GenericType())
            self._shape = (LiteralInteger(0),)
            self._is_homogeneous = False
            return

        # Get possible types of elements
        element_types = [a.class_type for a in args]
        # Create a set of types using the same key for compatible types
        unique_element_types = {((d.primitive_type, d.precision) if isinstance(d, FixedSizeNumericType) \
                                 else d) : d for d in element_types}

        self._shape = (LiteralInteger(len(args)),)

        if any(isinstance(d, SymbolicType) for d in unique_element_types):
            self._class_type = InhomogeneousTupleType(*[a.class_type for a in args])
            self._is_homogeneous = False
            return

        contains_pointers = any(isinstance(a, (Variable, IndexedElement)) and a.rank>0 and \
                            not isinstance(a.class_type, HomogeneousTupleType) for a in args)

        is_homogeneous = (not prefer_inhomogeneous) and len(unique_element_types) == 1 and \
                        not isinstance(args[0].class_type, InhomogeneousTupleType)
        if is_homogeneous and args[0].rank > 0:
            shapes = [tuple(None if isinstance(s, PyccelArrayShapeElement) else s for s in a.shape)
                        for a in args]
            if len(set(shapes)) > 1:
                is_homogeneous = False
            elif not contains_pointers:
                self._shape += args[0].shape

        self._is_homogeneous = is_homogeneous
        if is_homogeneous:
            if contains_pointers:
                self._class_type = InhomogeneousTupleType(*element_types)
            else:
                self._class_type = HomogeneousTupleType(unique_element_types.popitem()[1])

        else:
            self._class_type = InhomogeneousTupleType(*[a.class_type for a in args])

        assert self._class_type.shape_is_compatible(self._shape)

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

        Indicates whether all elements of the tuple have the same dtype,
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

    This class should not be used to create an instance, it is
    simply a place-holder to indicate the class to the semantic parser.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _static_type = TupleType

#==============================================================================
class PythonLen(PyccelFunction):
    """
    Represents a `len` expression in the code.

    Represents a call to the function `len` which calculates the length
    (aka the first element of the shape) of an object. This can usually
    be calculated in the generated code, but in an inhomogeneous object
    the integer value of the shape must be returned.

    If the shape is unknown and cannot be determined at compile time then
    the first element of the shape is a `PyccelArrayShapeElement` which
    can be used to generate the equivalent C code using the `STC` library.

    Parameters
    ----------
    arg : TypedAstNode
        The argument whose length is being examined.
    """
    __slots__ = ()

    def __new__(cls, arg):
        if isinstance(arg, LiteralString):
            return LiteralInteger(len(arg.python_value))
        elif isinstance(arg.class_type, StringType):
            return super().__new__(cls)
        else:
            return arg.shape[0]

    def __init__(self, arg):
        super().__init__(arg)

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
    __slots__ = ('_args', '_shape', '_class_type')
    _attribute_nodes = ('_args',)

    def __init__(self, *args):
        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        elif len(args) == 0:
            self._shape = (LiteralInteger(0),)
            self._class_type = HomogeneousListType(GenericType())
            return
        arg0 = args[0]
        is_homogeneous = arg0.class_type is not GenericType() and \
                         all(a.class_type is not GenericType() and \
                             arg0.class_type == a.class_type for a in args[1:])
        if is_homogeneous:
            dtype = arg0.class_type

            self._shape = (LiteralInteger(len(args)), )

        else:
            raise TypeError("Can't create an inhomogeneous list")

        self._class_type = HomogeneousListType(dtype)

    def __iter__(self):
        return self._args.__iter__()

    def __str__(self):
        args = ', '.join(str(a) for a in self)
        return f'[{args}]'

    def __repr__(self):
        args = ', '.join(str(a) for a in self)
        return f'PythonList({args})'

    def __len__(self):
        return len(self._args)

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

        Indicates whether all elements of the list have the same dtype,
        rank, etc (homogenous) or if these values can vary (inhomogeneous). Lists
        are always homogeneous.
        """
        return True


class PythonListFunction(PyccelFunction):
    """
    Class representing a call to the `list` function.

    Class representing a call to the `list` function. This is
    different to the `[,]` syntax as it only takes one argument
    and unpacks any variables.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function call.
    """
    name = 'list'
    __slots__ = ('_class_type', '_shape')
    _attribute_nodes = ()
    _static_type = HomogeneousListType

    def __new__(cls, arg = None):
        if arg is None:
            return PythonList()
        elif isinstance(arg.shape[0], LiteralInteger):
            return PythonList(*arg)
        else:
            return super().__new__(cls)

    def __init__(self, copied_obj):
        self._class_type = copied_obj.class_type
        self._shape = copied_obj.shape
        super().__init__(copied_obj)

    @property
    def copied_obj(self):
        """
        The object being copied.

        The object being copied to create a new list instance.
        """
        return self._args[0]


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
    __slots__ = ('_args','_class_type','_shape')
    _attribute_nodes = ('_args',)

    def __init__(self, *args):
        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        elif len(args) == 0:
            self._shape = (LiteralInteger(0),)
            self._class_type = HomogeneousSetType(GenericType())
            return

        arg0 = args[0]
        is_homogeneous = arg0.class_type is not GenericType() and \
                         all(a.class_type is not GenericType() and \
                             arg0.class_type == a.class_type for a in args[1:])
        if is_homogeneous:
            elem_type = arg0.class_type
            self._shape = (LiteralInteger(len(args)), )
            if elem_type.rank > 0:
                raise TypeError("Pyccel can't hash non-scalar types")
        else:
            raise TypeError("Can't create an inhomogeneous set")

        self._class_type = HomogeneousSetType(elem_type)

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

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

    def __str__(self):
        args = ', '.join(str(a) for a in self)
        return f'{{{args}}}'

    def __repr__(self):
        args = ', '.join(str(a) for a in self)
        return f'PythonSet({args})'


class PythonSetFunction(PyccelFunction):
    """
    Class representing a call to the `set` function.

    Class representing a call to the `set` function. This is
    different to the `{,}` syntax as it only takes one argument
    and unpacks any variables.

    Parameters
    ----------
    copied_obj : TypedAstNode
        The argument passed to the function call.
    """
    __slots__ = ('_shape', '_class_type')
    name = 'set'
    _static_type = HomogeneousSetType

    def __init__(self, copied_obj):
        self._class_type = copied_obj.class_type
        self._shape = copied_obj.shape
        super().__init__(copied_obj)

#==============================================================================
class PythonDict(PyccelFunction):
    """
    Class representing a call to Python's `{}` function.

    Class representing a call to Python's `{}` function which generates a
    literal Python dict. This operator does not handle `**a` expressions.

    Parameters
    ----------
    keys : iterable[TypedAstNode]
        The keys of the new dictionary.
    values : iterable[TypedAstNode]
        The values of the new dictionary.
    """
    __slots__ = ('_keys', '_values', '_shape', '_class_type')
    _attribute_nodes = ('_keys', '_values')
    _rank = 1

    def __init__(self, keys, values):
        self._keys = keys
        self._values = values
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        elif len(keys) != len(values):
            raise TypeError("Unpacking values in a dictionary is not yet supported.")
        elif len(keys) == 0:
            self._shape = (LiteralInteger(0),)
            self._class_type = DictType(GenericType(), GenericType())
            return

        key0 = keys[0]
        val0 = values[0]
        homogeneous_keys = all(k.class_type is not GenericType() for k in keys) and \
                           all(key0.class_type == k.class_type for k in keys[1:])
        homogeneous_vals = all(v.class_type is not GenericType() for v in values) and \
                           all(val0.class_type == v.class_type for v in values[1:])

        if homogeneous_keys and homogeneous_vals:
            self._class_type = DictType(key0.class_type, val0.class_type)

            self._shape = (LiteralInteger(len(keys)), )
        else:
            raise TypeError("Can't create an inhomogeneous dict")

    def __iter__(self):
        return zip(self._keys, self._values)

    def __str__(self):
        args = ', '.join(f'{k}: {v}' for k,v in self)
        return f'{{{args}}}'

    def __repr__(self):
        args = ', '.join(f'{repr(k)}: {repr(v)}' for k,v in self)
        return f'PythonDict({args})'

    def __len__(self):
        return len(self._keys)

    @property
    def keys(self):
        """
        The keys of the new dictionary.

        The keys of the new dictionary.
        """
        return self._keys

    @property
    def values(self):
        """
        The values of the new dictionary.

        The values of the new dictionary.
        """
        return self._values

#==============================================================================
class PythonDictFunction(PyccelFunction):
    """
    Class representing a call to the `dict` function.

    Class representing a call to the `dict` function. This is
    different to the `{}` syntax as it is either a cast function or it
    uses arguments to create the dictionary. In the case of a cast function
    an instance of PythonDict is returned to express this concept. In the
    case of a copy this class stores the description of the copy operator.

    Parameters
    ----------
    *args : TypedAstNode
        The arguments passed to the function call. If args are provided
        then only one argument should be provided. This object is copied
        unless it is a temporary PythonDict in which case it is returned
        directly.
    **kwargs : dict[TypedAstNode]
        The keyword arguments passed to the function call. If kwargs are
        provided then no args should be provided and a PythonDict object
        will be created.
    """
    __slots__ = ('_shape', '_class_type')
    name = 'dict'
    _static_type = DictType

    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            keys = [LiteralString(k) for k in kwargs]
            values = list(kwargs.values())
            return PythonDict(keys, values)
        elif len(args) != 1:
            raise NotImplementedError("Unrecognised dict calling convention")
        else:
            return super().__new__(cls)

    def __init__(self, copied_obj):
        self._class_type = copied_obj.class_type
        self._shape = copied_obj.shape
        super().__init__(copied_obj)

    @property
    def copied_obj(self):
        """
        The object being copied.

        The object being copied to create a new dict instance.
        """
        return self._args[0]

#==============================================================================
class PythonMap(Iterable):
    """
    Class representing a call to Python's builtin map function.

    Class representing a call to Python's builtin map function.

    Parameters
    ----------
    func : FunctionDef
        The function to be applied to the elements.

    func_args : TypedAstNode
        The arguments to which the function will be applied.
    """
    __slots__ = ('_func','_func_args')
    _attribute_nodes = Iterable._attribute_nodes + ('_func','_func_args')
    name = 'map'
    _class_type = SymbolicType()
    _shape = ()

    def __init__(self, func, func_args):
        self._func = func
        self._func_args = func_args
        super().__init__(1)

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

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        This is the function calling the element of the variable that is found
        at the index.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        idx = self._indices[0] if len(self._indices)==1 else self._indices
        return [self.func(IndexedElement(self.func_args, idx))]

    def get_range(self):
        """
        Get a range that can be used to iterate over the map iterable.

        Get a range that can be used to iterate over the map iterable.

        Returns
        -------
        PythonRange
            A range that can be used to iterate over the map iterable.
        """
        return PythonRange(PythonLen(self.func_args))

    def get_assign_targets(self):
        """
        Get objects that should be assigned to variables to use the map target.

        Get objects that should be assigned to variables to use the map targets.
        This is the function calling the element of the variable that is found
        at the index.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        return self.get_python_iterable_item()

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
class PythonRange(Iterable):
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
    _attribute_nodes = Iterable._attribute_nodes + ('_start', '_stop', '_step')
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
        assert self._stop is not None

        super().__init__(0)

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

    def get_range(self):
        """
        Get this range.

        Get this range. This method is used to allow this class to be handled
        like other iterables which can be converted to PythonRange objects.

        Returns
        -------
        PythonRange
            This object.
        """
        return self

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns an element of the range indexed with the iterators
        previously provided via the set_loop_counters method
        (useful to determine the dtype etc of the loop iterator).

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        return self._indices

    def get_assign_targets(self):
        """
        Get objects that should be assigned to variables to use the range.

        This method is used to allow this class to be handled like other iterables
        which can be converted to PythonRange objects.

        Returns
        -------
        list[TypedAstNode]
            An empty list.
        """
        return []

#==============================================================================
class PythonZip(Iterable):
    """
    Represents a call to Python `zip` for code generation.

    Represents a call to Python's built-in function `zip`.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ('_length', '_class_type','_args')
    _attribute_nodes = Iterable._attribute_nodes + ('_args', '_length')
    name = 'zip'

    def __init__(self, *args):
        if not isinstance(args, (tuple, list)):
            raise TypeError('args must be a list or tuple')
        elif len(args) < 2:
            raise ValueError('args must be of length > 2')
        self._args = args
        if pyccel_stage == 'syntactic':
            self._length = None
            return
        else:
            lengths = [a.shape[0].python_value for a in self.args if isinstance(a.shape[0], LiteralInteger)]
            if lengths:
                self._length = min(lengths)
            else:
                self._length = PythonMin(*[PythonLen(a) for a in self.args])
            self._class_type = InhomogeneousTupleType(*[a.class_type for a in args])
        super().__init__(1)

    @property
    def args(self):
        """
        The arguments passed to the function.

        Tuple containing all the arguments passed to the function call.
        """
        return self._args

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns an element of the zip indexed with the iterators
        previously provided via the set_loop_counters method
        (useful to determine the dtype etc of the loop iterator).

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        index = self._indices[0]
        return [a[index] for a in self.args]

    def get_assign_targets(self):
        """
        Get objects that should be assigned to variables to use the zip targets.

        Get objects that should be assigned to variables to use the zip targets.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        return self.get_python_iterable_item()

    def get_range(self):
        """
        Get a range that can be used to iterate over the zip iterable.

        Get a range that can be used to iterate over the zip iterable.

        Returns
        -------
        PythonRange
            A range that can be used to iterate over the zip iterable.
        """
        return PythonRange(self._length)

#==============================================================================
class PythonAbs(PyccelFunction):
    """
    Represents a call to Python `abs` for code generation.

    Represents a call to Python's built-in function `abs`.

    Parameters
    ----------
    x : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    name = 'abs'
    def __init__(self, x):
        self._shape     = x.shape
        self._class_type = PythonNativeInt() if x.dtype is PythonNativeInt() else PythonNativeFloat()
        super().__init__(x)

    @property
    def arg(self):
        """
        The argument passed to the abs function.

        The argument passed to the abs function.
        """
        return self._args[0]

#==============================================================================
class PythonSum(PyccelFunction):
    """
    Represents a call to Python `sum` for code generation.

    Represents a call to Python's built-in function `sum`.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_class_type',)
    name   = 'sum'
    _shape = None

    def __init__(self, arg):
        if not isinstance(arg, TypedAstNode):
            raise TypeError(f'Unknown type of {type(arg)}.' )
        if isinstance(arg.class_type, HomogeneousContainerType):
            self._class_type = arg.class_type.element_type
        else:
            self._class_type = sum(arg.class_type, start=GenericType())
        super().__init__(arg)

    @property
    def arg(self):
        """
        The argument passed to the sum function.

        The argument passed to the sum function.
        """
        return self._args[0]

#==============================================================================
class PythonMax(PyccelFunction):
    """
    Represents a call to Python's built-in `max` function.

    Represents a call to Python's built-in `max` function.

    Parameters
    ----------
    *x : list, tuple, PythonTuple, PythonList
        The arguments passed to the function.
    """
    __slots__ = ('_class_type',)
    name   = 'max'
    _shape = None

    def __init__(self, *x):
        if len(x)==1:
            x = x[0]

        if isinstance(x, (list, tuple)):
            x = PythonTuple(*x)
        elif not (isinstance(x, (PythonTuple, PythonList))
                  or (isinstance(x, Variable) and isinstance(x.class_type, HomogeneousContainerType))):
            raise TypeError(f'Unknown type of {type(x)}.' )

        if isinstance(x, (PythonTuple, PythonList)) and not x.is_homogeneous:
            types = ', '.join(str(xi.dtype) for xi in x)
            raise TypeError("Cannot determine final dtype of 'max' call with arguments of different "
                             f"types ({types}). Please cast arguments to the desired dtype")
        if isinstance(x.class_type, HomogeneousContainerType):
            self._class_type = x.class_type.element_type
        else:
            self._class_type = sum(x.class_type, start=GenericType())
        super().__init__(x)


#==============================================================================
class PythonMin(PyccelFunction):
    """
    Represents a call to Python's built-in `min` function.

    Represents a call to Python's built-in `min` function.

    Parameters
    ----------
    *x : list, tuple, PythonTuple, PythonList
        The arguments passed to the function.
    """
    __slots__ = ('_class_type',)
    name   = 'min'
    _shape = None

    def __init__(self, *x):
        if len(x)==1:
            x = x[0]

        if isinstance(x, (list, tuple)):
            x = PythonTuple(*x)
        elif not (isinstance(x, (PythonTuple, PythonList))
                  or (isinstance(x, Variable) and isinstance(x.class_type, HomogeneousContainerType))):
            raise TypeError(f'Unknown type of {type(x)}.' )

        if isinstance(x, (PythonTuple, PythonList)) and not x.is_homogeneous:
            types = ', '.join(str(xi.dtype) for xi in x)
            raise TypeError("Cannot determine final dtype of 'min' call with arguments of different "
                              f"types ({types}). Please cast arguments to the desired dtype")
        if isinstance(x.class_type, HomogeneousContainerType):
            self._class_type = x.class_type.element_type
        else:
            self._class_type = sum(x.class_type, start=GenericType())
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
        assert len(args) == len(self.variables)
        return self.expr.subs(self.variables, args)

    def __str__(self):
        return f"{self.variables} -> {self.expr}"

#==============================================================================
class PythonType(PyccelFunction):
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
    __slots__ = ('_type','_obj')
    _attribute_nodes = ('_obj',)
    _class_type = SymbolicType()
    _shape = None
    _static_type = TypeAlias()

    def __init__(self, obj):
        if not isinstance (obj, TypedAstNode):
            raise TypeError(f"Python's type function is not implemented for {type(obj)} object")
        self._type = obj.class_type
        self._obj = obj

        super().__init__()

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
        return LiteralString(f"<class '{self._type}'>")

#==============================================================================
class VariableIterator(Iterable):
    """
    Represents a call to Python's `iter` function on a variable.

    Represents a call to Python's `iter` function on a variable. This is
    useful for for loops as this function is called implicitly.

    Parameters
    ----------
    var : Variable
        The Variable that is iterated over.
    """
    __slots__ = ('_var',)
    _attribute_nodes = Iterable._attribute_nodes + ('_var',)

    def __init__(self, var):
        assert isinstance(var, TypedAstNode)
        assert var.class_type is not VoidType()
        assert var.rank > 0
        self._var = var
        super().__init__(1)

    @property
    def variable(self):
        """
        The variable being iterated over.

        The variable being iterated over.
        """
        return self._var

    def get_range(self):
        """
        Get a range that can be used to iterate over the variable.

        Get a range that can be used to iterate over the variable.

        Returns
        -------
        PythonRange
            A range that can be used to iterate over the enumerate iterable.
        """
        return PythonRange(self.variable.shape[0])

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        Returns an element of the variable indexed with the iterators
        previously provided via the set_loop_counters method
        (useful to determine the dtype etc of the loop iterator).

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        return [self._var[self._indices[0]]]

    def get_assign_targets(self):
        """
        Get objects that should be assigned to variables to use the variable iterator targets.

        Returns an element of the variable indexed with the iterators
        previously provided via the set_loop_counters method.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        return self.get_python_iterable_item()

#==============================================================================
class PythonIsInstance(PyccelFunction):
    """
    Represents a call to Python's `isinstance` function.

    Represents a call to Python's `isinstance` function which checks if an
    object has a specified type. This class exists to find a definition of
    `isinstance` in builtin_functions_dict but it should not be instantiated.

    Parameters
    ----------
    obj : TypedAstNode
        The object whose type should be checked.
    class_or_tuple : TypedAstNode
        A class or a tuple of classes describing the acceptable types for
        the object.
    """
    __slots__ = ()
    def __init__(self, obj, class_or_tuple):
        super().__init__(obj, class_or_tuple)

#==============================================================================
class PythonStr(PyccelFunction):
    """
    Represents a call to Python's `str` function.

    Represents a call to Python's `str` function which describes a string
    cast.

    Parameters
    ----------
    arg : TypedAstNode
        The argument that is cast to a string.
    """
    __slots__ = ('_shape',)
    _static_type = StringType()
    _class_type = StringType()
    name = 'str'

    def __new__(cls, arg):
        if isinstance(arg, LiteralString):
            return arg
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        if not isinstance(arg.class_type, (StringType, CharType)):
            raise NotImplementedError("Support for casting non-character types to strings is not yet available")
        self._shape = (None,)
        super().__init__(arg)

#==============================================================================

DtypePrecisionToCastFunction = {
        PythonNativeBool()    : PythonBool,
        PythonNativeInt()     : PythonInt,
        PythonNativeFloat()   : PythonFloat,
        PythonNativeComplex() : PythonComplex,
}

#==============================================================================

builtin_functions_dict = {
    'abs'        : PythonAbs,
    'bool'       : PythonBool,
    'complex'    : PythonComplex,
    'dict'       : PythonDictFunction,
    'enumerate'  : PythonEnumerate,
    'float'      : PythonFloat,
    'int'        : PythonInt,
    'isinstance' : PythonIsInstance,
    'len'        : PythonLen,
    'list'       : PythonListFunction,
    'map'        : PythonMap,
    'max'        : PythonMax,
    'min'        : PythonMin,
    'not'        : PyccelNot,
    'range'      : PythonRange,
    'round'      : PythonRound,
    'set'        : PythonSetFunction,
    'str'        : PythonStr,
    'sum'        : PythonSum,
    'tuple'      : PythonTupleFunction,
    'type'       : PythonType,
    'zip'        : PythonZip,
}

original_type_to_pyccel_type[list] = PythonListFunction
original_type_to_pyccel_type[set] = PythonSetFunction
original_type_to_pyccel_type[dict] = PythonDictFunction
original_type_to_pyccel_type[tuple] = PythonTupleFunction
