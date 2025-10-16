#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the numpy module understood by pyccel
"""
from packaging.version import Version

import numpy

from pyccel.errors.errors import Errors
from pyccel.errors.messages import WRONG_LINSPACE_ENDPOINT, NON_LITERAL_KEEP_DIMS, NON_LITERAL_AXIS

from pyccel.utilities.stage import PyccelStage

from .basic          import TypedAstNode
from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList,
                             PythonType, PythonConjugate, DtypePrecisionToCastFunction)

from .core           import Module, Import, PyccelFunctionDef, FunctionCall

from .datatypes      import PythonNativeBool, PythonNativeInt, PythonNativeFloat
from .datatypes      import PrimitiveBooleanType, PrimitiveIntegerType, PrimitiveFloatingPointType, PrimitiveComplexType
from .datatypes      import HomogeneousTupleType, FixedSizeNumericType, GenericType
from .datatypes      import InhomogeneousTupleType, ContainerType, SymbolicType

from .internals      import PyccelFunction, Slice
from .internals      import PyccelArraySize, PyccelArrayShapeElement

from .literals       import LiteralInteger, LiteralString, convert_to_literal
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .numpytypes     import NumpyNumericType, NumpyInt8Type, NumpyInt16Type, NumpyInt32Type, NumpyInt64Type
from .numpytypes     import NumpyFloat32Type, NumpyFloat64Type, NumpyFloat128Type, NumpyNDArrayType
from .numpytypes     import NumpyComplex64Type, NumpyComplex128Type, NumpyComplex256Type, numpy_precision_map
from .operators      import broadcast, PyccelMinus, PyccelDiv, PyccelMul, PyccelAdd
from .type_annotations import VariableTypeAnnotation, typenames_to_dtypes as dtype_registry
from .variable       import Variable, Constant, IndexedElement

errors = Errors()
pyccel_stage = PyccelStage()
numpy_v2_1 = Version(numpy.__version__) >= Version("2.1.0")

__all__ = (
    'process_shape',
    # --- Base classes ---
    'NumpyAutoFill',
    'NumpyNewArray',
    'NumpyUfuncBase',
    'NumpyUfuncBinary',
    'NumpyUfuncUnary',
    # --- Array allocation ---
    'NumpyArange',
    'NumpyArray',
    'NumpyEmpty',
    'NumpyEmptyLike',
    'NumpyFull',
    'NumpyFullLike',
    'NumpyLinspace',
    'NumpyNonZeroElement',
    'NumpyOnes',
    'NumpyOnesLike',
    'NumpyZeros',
    'NumpyZerosLike',
    # --- Unary functions ---
    'NumpyAbs',
    'NumpyArccos',
    'NumpyArccosh',
    'NumpyArcsin',
    'NumpyArcsinh',
    'NumpyArctan',
    'NumpyArctan2',
    'NumpyArctanh',
    'NumpyCos',
    'NumpyCosh',
    'NumpyDivide',
    'NumpyExp',
    'NumpyExpm1',
    'NumpyFabs',
    'NumpyFloor',
    'NumpyHypot',
    'NumpyIsFinite',
    'NumpyIsInf',
    'NumpyIsNan',
    'NumpyLog',
    'NumpySign',
    'NumpySin',
    'NumpySinh',
    'NumpySqrt',
    'NumpyTan',
    'NumpyTanh',
    'NumpyTranspose',
    # --- Cast methods ---
    'NumpyBool',
    'NumpyComplex',
    'NumpyComplex64',
    'NumpyComplex128',
    'NumpyFloat',
    'NumpyFloat32',
    'NumpyFloat64',
    'NumpyInt',
    'NumpyInt8',
    'NumpyInt16',
    'NumpyInt32',
    'NumpyInt64',
    # --- Other NumPy functions ---
    'NumpyAmax',
    'NumpyAmin',
    'NumpyConjugate',
    'NumpyCountNonZero',
    'NumpyImag',
    'NumpyMatmul',
    'NumpyMod',
    'NumpyNDArray',
    'NumpyNonZero',
    'NumpyNorm',
    'NumpyProduct',
    'NumpyRand',
    'NumpyRandint',
    'NumpyReal',
    'NumpyResultType',
    'NumpyShape',
    'NumpySize',
    'NumpySum',
    'NumpyWhere',
)

dtype_registry.update({
    'int8'       : NumpyInt8Type(),
    'int16'      : NumpyInt16Type(),
    'int32'      : NumpyInt32Type(),
    'int64'      : NumpyInt64Type(),
    'i1'         : NumpyInt8Type(),
    'i2'         : NumpyInt16Type(),
    'i4'         : NumpyInt32Type(),
    'i8'         : NumpyInt64Type(),
    'float32'    : NumpyFloat32Type(),
    'float64'    : NumpyFloat64Type(),
    'float128'   : NumpyFloat128Type(),
    'f4'         : NumpyFloat32Type(),
    'f8'         : NumpyFloat64Type(),
    'complex64'  : NumpyComplex64Type(),
    'complex128' : NumpyComplex128Type(),
    'complex256' : NumpyComplex256Type(),
    'c8'         : NumpyComplex64Type(),
    'c16'        : NumpyComplex128Type(),
    })

#=======================================================================================
def get_shape_of_multi_level_container(expr, shape_prefix = ()):
    """
    Get the shape of a multi-level container.

    Get the shape of a multi-level container such as a list of list
    of tuple of tuple. These objects only store the shape of the top
    layer as the elements may have different sizes but in NumPy
    functions we can assume that the shape is homogeneous as otherwise
    the original Python code would raise errors.

    Parameters
    ----------
    expr : TypedAstNode
        The expression whose shape we want to know.
    shape_prefix : tuple[TypedAstNode, ...], optional
        A tuple of objects describing the shape of the containers where
        the expression is found. This is used internally to call this
        function recursively. In most cases it is not necessary to
        provide this value when calling this function.

    Returns
    -------
    tuple[TypedAstNode, ...]
        A tuple of objects describing the shape of the mult-level container.
    """
    class_type = expr.class_type
    assert isinstance(class_type, ContainerType)

    shape = expr.shape
    new_shape = shape_prefix + expr.shape

    assert len(shape) <= class_type.rank

    if class_type.rank == len(shape):
        return new_shape
    elif isinstance(expr, (PythonTuple, PythonList)):
        return get_shape_of_multi_level_container(expr.args[0], new_shape)
    elif isinstance(expr, (Variable, IndexedElement)):
        return get_shape_of_multi_level_container(expr[LiteralInteger(0)], new_shape)
    else:
        errors.report(f"Can't calculate shape of object of type {type(expr)}",
                severity = 'error', symbol=expr)
        return (None,)*expr.rank

#=======================================================================================
def process_shape(is_scalar, shape):
    """
    Modify the input shape to the expected type.

    Modify the input shape to the expected type.

    Parameters
    ----------
    is_scalar : bool
        True if the result is a scalar, False if it is an array.
    shape : TypedAstNode | iterable | int
        Input shape.

    Returns
    -------
    tuple[int | TypedAstNode]
        The shape of the array in a compatible format.
    """
    if is_scalar:
        return None
    elif shape is None:
        return ()
    elif isinstance(shape, TypedAstNode):
        if shape.rank == 0:
            shape = [shape]
    elif not hasattr(shape, '__iter__'):
        shape = [shape]

    new_shape = []
    for s in shape:
        if isinstance(s,(LiteralInteger, Variable, Slice, TypedAstNode, FunctionCall)):
            new_shape.append(s)
        elif isinstance(s, int):
            new_shape.append(LiteralInteger(s))
        else:
            raise TypeError('shape elements cannot be '+str(type(s))+'. They must be one of the following types: LiteralInteger, Variable, Slice, TypedAstNode, int, FunctionCall')
    return tuple(new_shape)

#=======================================================================================
class NumpyFloat(PythonFloat):
    """
    Represents a call to `numpy.float()` function.

    Represents a call to the NumPy cast function `float`.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    _static_type = NumpyFloat64Type()
    name = 'float'

    def __init__(self, arg):
        self._shape = arg.shape
        rank  = arg.rank
        order = arg.order
        self._class_type = NumpyNDArrayType(self.static_type(), rank, order) if rank else self.static_type()
        super().__init__(arg)

    @property
    def is_elemental(self):
        """
        Indicates whether the function can be applied elementwise.
        
        Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

class NumpyFloat32(NumpyFloat):
    """
    Represents a call to numpy.float32() function.

    Represents a call to numpy.float32() function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    _static_type = NumpyFloat32Type()
    name = 'float32'

class NumpyFloat64(NumpyFloat):
    """
    Represents a call to numpy.float64() function.

    Represents a call to numpy.float64() function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    _static_type = NumpyFloat64Type()
    name = 'float64'

#=======================================================================================
class NumpyBool(PythonBool):
    """
    Represents a call to `numpy.bool()` function.

    Represents a call to the NumPy cast function `bool`.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    name = 'bool'
    def __init__(self, arg):
        self._shape = arg.shape
        rank  = arg.rank
        order = arg.order
        self._class_type = NumpyNDArrayType(self.static_type(), rank, order) if rank else self.static_type()
        super().__init__(arg)

    @property
    def is_elemental(self):
        """
        Indicates whether the function can be applied elementwise.
        
        Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#=======================================================================================
# TODO [YG, 13.03.2020]: handle case where base != 10
class NumpyInt(PythonInt):
    """
    Represents a call to `numpy.int()` function.

    Represents a call to the NumPy cast function `int`.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    base : TypedAstNode
        The argument passed to the function to indicate the base in which
        the integer is expressed.
    """
    __slots__ = ('_shape','_class_type')
    _static_type = numpy_precision_map[(PrimitiveIntegerType(), PythonInt._static_type.precision)]
    name = 'int'

    def __init__(self, arg=None, base=10):
        if base != 10:
            raise TypeError("numpy.int's base argument is not yet supported")
        self._shape = arg.shape
        rank  = arg.rank
        order = arg.order
        self._class_type = NumpyNDArrayType(self.static_type(), rank, order) if rank else self.static_type()
        super().__init__(arg)

    @property
    def is_elemental(self):
        """
        Indicates whether the function can be applied elementwise.
        
        Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

class NumpyInt8(NumpyInt):
    """
    Represents a call to numpy.int8() function.

    Represents a call to numpy.int8() function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    _static_type = NumpyInt8Type()
    name = 'int8'

class NumpyInt16(NumpyInt):
    """
    Represents a call to numpy.int16() function.

    Represents a call to numpy.int16() function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    _static_type = NumpyInt16Type()
    name = 'int16'

class NumpyInt32(NumpyInt):
    """
    Represents a call to numpy.int32() function.

    Represents a call to numpy.int32() function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    _static_type = NumpyInt32Type()
    name = 'int32'

class NumpyInt64(NumpyInt):
    """
    Represents a call to numpy.int64() function.

    Represents a call to numpy.int64() function.

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    _static_type = NumpyInt64Type()
    name = 'int64'

#==============================================================================
class NumpyReal(PythonReal):
    """
    Represents a call to numpy.real for code generation.

    Represents a call to the NumPy function real.
    > a = 1+2j
    > np.real(a)
    1.0

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    name = 'real'
    def __new__(cls, arg):
        if isinstance(arg.dtype, PythonNativeBool):
            if arg.rank:
                return NumpyInt(arg)
            else:
                return PythonInt(arg)
        else:
            return super().__new__(cls, arg)

    def __init__(self, arg):
        super().__init__(arg)
        rank  = arg.rank
        order = arg.order
        dtype = arg.dtype.element_type
        self._class_type = NumpyNDArrayType(dtype, rank, order) if rank else dtype
        self._shape = process_shape(self.rank == 0, self.internal_var.shape)

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#==============================================================================

class NumpyImag(PythonImag):
    """
    Represents a call to numpy.imag for code generation.

    Represents a call to the NumPy function imag.
    > a = 1+2j
    > np.imag(a)
    2.0

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    name = 'imag'
    def __new__(cls, arg):

        if not isinstance(arg.dtype.primitive_type, PrimitiveComplexType):
            dtype = PythonNativeInt() if isinstance(arg.dtype, PythonNativeBool) else arg.dtype
            if arg.rank == 0:
                return convert_to_literal(0, dtype)
            dtype = DtypePrecisionToCastFunction[dtype].static_type()
            return NumpyZeros(arg.shape, dtype=dtype)
        return super().__new__(cls, arg)

    def __init__(self, arg):
        super().__init__(arg)
        rank  = arg.rank
        order = arg.order
        dtype = arg.dtype.element_type
        self._class_type = NumpyNDArrayType(dtype, rank, order) if rank else dtype
        self._shape = process_shape(self.rank == 0, self.internal_var.shape)

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#=======================================================================================
class NumpyComplex(PythonComplex):
    """
    Represents a call to `numpy.complex()` function.

    Represents a call to the NumPy cast function `complex`.

    Parameters
    ----------
    arg0 : TypedAstNode
        The first argument passed to the function. Either the array/scalar being cast
        or the real part of the complex.
    arg1 : TypedAstNode, optional
        The second argument passed to the function. The imaginary part of the complex.
    """
    _real_cast = NumpyReal
    _imag_cast = NumpyImag
    __slots__ = ('_shape','_class_type')
    _static_type = NumpyComplex128Type()
    name = 'complex'

    def __init__(self, arg0, arg1 = None):
        if arg1 is not None:
            raise NotImplementedError("Use builtin complex function not deprecated np.complex")
        self._shape = arg0.shape
        rank  = arg0.rank
        order = arg0.order
        self._class_type = NumpyNDArrayType(self.static_type(), rank, order) if rank else self.static_type()
        super().__init__(arg0)

    @property
    def is_elemental(self):
        """
        Indicates whether the function can be applied elementwise.
        
        Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

class NumpyComplex64(NumpyComplex):
    """
    Represents a call to numpy.complex64() function.

    Represents a call to numpy.complex64() function.

    Parameters
    ----------
    arg0 : TypedAstNode
        The argument passed to the function.

    arg1 : TypedAstNode
        Unused inherited argument.
    """
    __slots__ = ()
    _static_type = NumpyComplex64Type()
    name = 'complex64'

class NumpyComplex128(NumpyComplex):
    """
    Represents a call to numpy.complex128() function.

    Represents a call to numpy.complex128() function.

    Parameters
    ----------
    arg0 : TypedAstNode
        The argument passed to the function.

    arg1 : TypedAstNode
        Unused inherited argument.
    """
    __slots__ = ()
    _static_type = NumpyComplex128Type()
    name = 'complex128'

#=======================================================================================

class NumpyResultType(PyccelFunction):
    """
    Class representing a call to the `numpy.result_type` function.

    A class representing a call to the NumPy function `result_type` which returns
    the datatype of an expression. This function can be used to access the `dtype`
    property of a NumPy array.

    Parameters
    ----------
    *arrays_and_dtypes : TypedAstNode
        Any arrays and dtypes passed to the function (currently only accepts one array
        and no dtypes).
    """
    __slots__ = ('_class_type',)
    _shape = None
    name = 'result_type'

    def __init__(self, *arrays_and_dtypes):
        types = [a.cls_name.static_type() if isinstance(a, PyccelFunctionDef) else a.class_type for a in arrays_and_dtypes]
        self._class_type = sum(types, start=GenericType())
        if isinstance(self._class_type, ContainerType):
            self._class_type = self._class_type.element_type

        super().__init__(*arrays_and_dtypes)

#==============================================================================

def process_dtype(dtype):
    """
    Analyse a dtype passed to a NumPy array creation function.

    This function takes a dtype passed to a NumPy array creation function,
    processes it in different ways depending on its type, and finally extracts
    the corresponding type and precision from the `dtype_registry` dictionary.

    This function could be useful when working with numpy creation function
    having a dtype argument, like numpy.array, numpy.arrange, numpy.linspace...

    Parameters
    ----------
    dtype : PythonType, PyccelFunctionDef, LiteralString, str, VariableTypeAnnotation
        The actual dtype passed to the NumPy function.

    Returns
    -------
    Datatype
        The Datatype corresponding to the passed dtype.
    int
        The precision corresponding to the passed dtype.

    Raises
    ------
    TypeError: In the case of unrecognized argument type.
    TypeError: In the case of passed string argument not recognized as valid dtype.
    """
    if isinstance(dtype, VariableTypeAnnotation):
        dtype = dtype.class_type

    if isinstance(dtype, PythonType):
        if dtype.arg.rank > 0:
            errors.report("Python's type function doesn't return enough information about this object for pyccel to fully define a type",
                    symbol=dtype, severity="fatal")
        else:
            dtype = dtype.arg.class_type
    elif isinstance(dtype, NumpyResultType):
        dtype =  dtype.dtype

    elif isinstance(dtype, PyccelFunctionDef):
        dtype = dtype.cls_name.static_type()

    elif isinstance(dtype, (LiteralString, str)):
        try:
            dtype = dtype_registry[str(dtype)]
        except KeyError:
            raise TypeError(f'Unknown type of {dtype}.')

    if isinstance(dtype, (NumpyNumericType, PythonNativeBool, GenericType)):
        return dtype
    if isinstance(dtype, FixedSizeNumericType):
        return numpy_precision_map[(dtype.primitive_type, dtype.precision)]
    else:
        raise TypeError(f'Unknown type of {dtype}.')

#==============================================================================
class NumpyNewArray(PyccelFunction):
    """
    Superclass for nodes representing NumPy array allocation functions.

    Class from which all nodes representing a NumPy function which implies a call
    to `Allocate` should inherit.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments of the superclass PyccelFunction.
    class_type : NumpyNDArrayType
        The type of the new array.
    init_dtype : PythonType, PyccelFunctionDef, LiteralString, str
        The actual dtype passed to the NumPy function.
    """
    __slots__ = ('_init_dtype','_class_type')

    def __init__(self, *args, class_type, init_dtype = None):
        assert isinstance(class_type, NumpyNDArrayType)
        self._init_dtype = init_dtype
        self._class_type = class_type # pylint: disable=no-member

        super().__init__(*args)

    @property
    def init_dtype(self):
        """
        The dtype provided to the function when it was initialised in Python.

        The dtype provided to the function when it was initialised in Python.
        If no dtype was provided then this should equal `None`.
        """
        return self._init_dtype

    #--------------------------------------------------------------------------
    @staticmethod
    def _process_order(rank, order):
        """
        Treat the order to get an order in the format expected by Pyccel.

        Process the order passed to the array creation function to get an order
        in the format expected by Pyccel. The final format should be a string
        containing either 'C' or 'F'.

        Parameters
        ----------
        rank : int
            The rank of the array being created.
        order : str | LiteralString
            The order of the array as specified by the user or the subclass.

        Returns
        -------
        str | None
            The order in the format expected by Pyccel.
        """

        if rank < 2:
            return None

        order = str(order).strip('\'"')
        assert order in ('C', 'F')
        return order

#==============================================================================
class NumpyArray(NumpyNewArray):
    """
    Represents a call to `numpy.array` for code generation.

    A class representing a call to the NumPy `array` function.

    Parameters
    ----------
    arg : list, tuple, PythonList
        The data from which the array is initialised.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str
        The data type passed to the NumPy function.

    order : str
        The ordering of the array (C/Fortran).

    ndmin : LiteralInteger, int, optional
        The minimum number of dimensions that the resulting array should
        have.
    """
    __slots__ = ('_arg','_shape')
    _attribute_nodes = NumpyNewArray._attribute_nodes + ('_arg',)
    name = 'array'

    def __init__(self, arg, dtype=None, order='K', ndmin=None):

        assert isinstance(arg, (PythonTuple, PythonList, Variable, IndexedElement))
        assert isinstance(order, str)

        init_dtype = dtype

        if isinstance(arg.class_type, InhomogeneousTupleType):
            # If pseudo-inhomogeneous due to pointers, extract underlying dtype
            if dtype is None:
                dtype = arg[0].class_type.datatype
            dtype = process_dtype(dtype)
        else:
            # Verify dtype and get precision
            if dtype is None:
                dtype = arg.dtype
            dtype = process_dtype(dtype)

        shape = process_shape(False, get_shape_of_multi_level_container(arg))

        rank  = arg.rank
        assert len(shape) == rank

        if ndmin and ndmin>rank:
            shape = (LiteralInteger(1),)*(ndmin-rank) + shape
            rank = ndmin

        if rank < 2:
            order = None
        else:
            # ... Determine ordering
            order = str(order).strip("\'")

            assert order in ('K', 'A', 'C', 'F')

            if order in ('K', 'A'):
                order = arg.order or 'C'
            # ...

        self._arg   = arg
        self._shape = shape
        super().__init__(class_type = NumpyNDArrayType(dtype, rank, order), init_dtype = init_dtype)

    def __str__(self):
        return str(self.arg)

    @property
    def arg(self):
        """
        The data from which the array is initialised.

        A PyccelAstNode describing the data from which the array is initialised.
        """
        return self._arg

#==============================================================================
class NumpyArange(NumpyNewArray):
    """
    Represents a call to  numpy.arange for code generation.

    A class representing a call to the NumPy `arange` function.

    Parameters
    ----------
    start : Numeric
        Start of interval, default value 0.

    stop : Numeric
        End of interval.

    step : Numeric
        Spacing between values, default value 1.

    dtype : Datatype
        The type of the output array, if dtype is not given,
        infer the data type from the other input arguments.
    """
    __slots__ = ('_start','_step','_stop','_shape')
    _attribute_nodes = ('_start','_step','_stop')
    name = 'arange'

    def __init__(self, start, stop = None, step = None, dtype = None):

        if stop is None:
            self._start = LiteralInteger(0)
            self._stop = start
        else:
            self._start = start
            self._stop = stop
        self._step = step if step is not None else LiteralInteger(1)

        init_dtype = dtype
        if dtype is None:
            type_info = NumpyResultType(*self.arg)
            dtype = type_info.dtype

        self._shape = (MathCeil(PyccelDiv(PyccelMinus(self._stop, self._start), self._step)))
        self._shape = process_shape(False, self._shape)
        dtype = process_dtype(dtype)
        super().__init__(class_type = NumpyNDArrayType(dtype, 1, None), init_dtype = init_dtype)

    @property
    def arg(self):
        return (self._start, self._stop, self._step)

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
        step = PyccelMul(index, self.step, simplify=True)
        return PyccelAdd(self.start, step, simplify=True)

#==============================================================================
class NumpySum(PyccelFunction):
    """
    Represents a call to  numpy.sum for code generation.

    Represents a call to  numpy.sum for code generation.

    Parameters
    ----------
    arg : list , tuple , PythonTuple, PythonList, Variable
        The argument passed to the sum function.
    """
    __slots__ = ('_class_type',)
    name = 'sum'
    _shape = None

    def __init__(self, arg):
        if not isinstance(arg, TypedAstNode):
            raise TypeError(f'Unknown type of {type(arg)}.')
        super().__init__(arg)
        lowest_possible_type = process_dtype(PythonNativeInt())
        if isinstance(arg.dtype.primitive_type, (PrimitiveBooleanType, PrimitiveIntegerType)) and \
                arg.dtype.precision <= lowest_possible_type.precision:
            self._class_type = lowest_possible_type
        else:
            self._class_type = process_dtype(arg.dtype)

    @property
    def arg(self):
        return self._args[0]

#==============================================================================
class NumpyProduct(PyccelFunction):
    """
    Represents a call to numpy.prod for code generation.

    Represents a call to numpy.prod for code generation.

    Parameters
    ----------
    arg : list , tuple , PythonTuple, PythonList, Variable
        The argument passed to the prod function.
    """
    __slots__ = ('_arg','_class_type')
    name = 'product'
    _shape = None

    def __init__(self, arg):
        if not isinstance(arg, TypedAstNode):
            raise TypeError(f'Unknown type of {type(arg)}.')
        super().__init__(arg)
        self._arg = PythonTuple(arg) if arg.rank == 0 else self._args[0]
        lowest_possible_type = process_dtype(PythonNativeInt())
        if isinstance(arg.dtype.primitive_type, (PrimitiveBooleanType, PrimitiveIntegerType)) and \
                arg.dtype.precision <= lowest_possible_type.precision:
            self._class_type = lowest_possible_type
        else:
            self._class_type = process_dtype(arg.dtype)

        default_cast = DtypePrecisionToCastFunction[self._class_type]
        self._arg = default_cast(self._arg) if arg.dtype != self._class_type else self._arg

    @property
    def arg(self):
        return self._arg


#==============================================================================
class NumpyMatmul(PyccelFunction):
    """
    Represents a call to numpy.matmul for code generation.

    Represents a call to NumPy's `matmul` function for code generation.

    Parameters
    ----------
    a : TypedAstNode
        The first argument of the matrix multiplication.
    b : TypedAstNode
        The second argument of the matrix multiplication.
    """
    __slots__ = ('_shape','_class_type')
    name = 'matmul'

    def __init__(self, a ,b):
        super().__init__(a, b)
        if pyccel_stage == 'syntactic':
            return

        if not isinstance(a, TypedAstNode):
            raise TypeError(f'Unknown type of {type(a)}.')
        if not isinstance(b, TypedAstNode):
            raise TypeError(f'Unknown type of {type(a)}.')

        args      = (a, b)
        type_info = NumpyResultType(*args)
        dtype = process_dtype(type_info.dtype)

        if not (a.shape is None or b.shape is None):

            m = 1 if a.rank < 2 else a.shape[0]
            n = 1 if b.rank < 2 else b.shape[1]
            self._shape = (m, n)

        if a.rank == 1 and b.rank == 1:
            rank  = 0
            self._shape = None
        elif a.rank == 1 or b.rank == 1:
            rank  = 1
            self._shape = (b.shape[1] if a.rank == 1 else a.shape[0],)
        else:
            rank = 2


        if a.order == b.order:
            order = a.order
        else:
            order = None if rank < 2 else 'C'

        self._class_type = NumpyNDArrayType(dtype, rank, order) if rank else dtype

    @property
    def a(self):
        return self._args[0]

    @property
    def b(self):
        return self._args[1]

#==============================================================================
class NumpyShape(PyccelFunction):
    """
    Represents a call to numpy.shape for code generation.

    This wrapper class represents calls to the function `numpy.shape` in the
    user code, or equivalently to the `shape` property of a `numpy.ndarray`.

    Objects of this class are never present in the Pyccel AST, because the
    class constructor always returns a PythonTuple with the required shape.

    Parameters
    ----------
    arg : TypedAstNode
        The Numpy array whose shape is being investigated.

    Returns
    -------
    PythonTuple
        The shape of the Numpy array, i.e. its size along each dimension.
    """

    __slots__ = ()
    name = 'shape'

    def __new__(cls, arg):
        return PythonTuple(*get_shape_of_multi_level_container(arg))

#==============================================================================
class NumpyLinspace(NumpyNewArray):

    """
    Represents a call to the function `numpy.linspace`.

    A class representing a call to the NumPy `linspace` function which returns `num`
    evenly spaced samples, calculated over the interval [start, stop].

    Parameters
    ----------
    start : list , tuple , PythonTuple, PythonList, Variable, Literals
         Represents the starting value of the sequence.

    stop : list , tuple , PythonTuple, PythonList, Variable, Literals
         Represents the ending value of the sequence (if endpoint is set to False).

    num : int, optional
         Number of samples to generate. Default is 50. Must be non-negative.

    endpoint : bool, optional
         If True, stop is the last sample. Otherwise, it is not included. Default is True.

    dtype : str, PyccelType
         The type of the output array. If dtype is not given, the data type is calculated
         from start and stop, the calculated dtype will never be an integer.
    """

    __slots__ = ('_index','_start','_stop', '_num','_endpoint','_shape', '_ind',
            '_step', '_py_argument')
    _attribute_nodes = ('_start', '_stop', '_index', '_step', '_num',
            '_endpoint', '_ind')
    name = 'linspace'

    def __init__(self, start, stop, num=None, endpoint=True, dtype=None):

        if not num:
            num = LiteralInteger(50)

        if num.rank != 0 or not isinstance(getattr(num.dtype, 'primitive_type', None), PrimitiveIntegerType):
            raise TypeError('Expecting positive integer num argument.')

        if any(not isinstance(arg, TypedAstNode) for arg in (start, stop, num)):
            raise TypeError('Expecting valid args.')

        init_dtype = dtype
        if dtype:
            final_dtype = process_dtype(dtype)
        else:
            args      = (start, stop)
            type_info = NumpyResultType(*args)
            if type_info.dtype.primitive_type is PrimitiveIntegerType():
                final_dtype = NumpyFloat64Type()
            else:
                final_dtype = process_dtype(type_info.dtype)

        self._index = Variable(PythonNativeInt(), 'linspace_index')
        self._start = start
        self._stop  = stop
        self._num  = num
        if endpoint is True:
            self._endpoint = LiteralTrue()
        elif endpoint is False:
            self._endpoint = LiteralFalse()
        else:
            if not isinstance(endpoint.dtype, PythonNativeBool):
                errors.report(WRONG_LINSPACE_ENDPOINT, symbol=endpoint, severity="fatal")
            self._endpoint = endpoint

        shape = broadcast(self._start.shape, self._stop.shape)
        self._shape = (self._num,)
        if shape is not None:
            self._shape += shape
        rank  = len(self._shape)
        order = None if rank < 2 else 'C'

        self._ind = None

        if isinstance(self.endpoint, LiteralFalse):
            self._step = PyccelDiv(PyccelMinus(self._stop, self._start), self.num)
        elif isinstance(self.endpoint, LiteralTrue):
            self._step = PyccelDiv(PyccelMinus(self._stop, self._start), PyccelMinus(self.num, LiteralInteger(1), simplify=True))
        else:
            self._step = PyccelDiv(PyccelMinus(self.stop, self.start), PyccelMinus(self.num, PythonInt(self.endpoint)))

        class_type = NumpyNDArrayType(final_dtype, rank, order)

        super().__init__(class_type = class_type, init_dtype = init_dtype)

    @property
    def endpoint(self):
        """Indicates if the stop must be included or not."""
        return self._endpoint

    @property
    def start(self):
        """Represent the starting value of the sequence."""
        return self._start

    @property
    def stop(self):
        """Represent the end value of the sequence, if the endpoint is False the stop will not be included."""
        return self._stop

    @property
    def num(self):
        """Represent the number of generated elements by the linspace function."""
        return self._num

    @property
    def index(self):
        """Used in the fortran codegen when there is no for loop created."""
        return self._index

    @property
    def step(self):
        """Represent size of spacing between generated elements."""
        return self._step

    @property
    def ind(self):
        """Used to store the index generated by the created for loop and needed by linspace function."""
        return self._ind

    @ind.setter
    def ind(self, value):
        assert self._ind is None
        value.set_current_user_node(self)
        self._ind = value

    @property
    def is_elemental(self):
        return True

#==============================================================================
class NumpyWhere(PyccelFunction):
    """
    Represents a call to `numpy.where`.

    Represents a call to NumPy's `where` function.

    Parameters
    ----------
    condition : TypedAstNode
        The condition which determines which value is returned.

    x : TypedAstNode, optional
        The value if True. If `x` is provided, `y` should also be provided.

    y : TypedAstNode, optional
        The value if False. If `y` is provided, `x` should also be provided.
    """

    __slots__ = ('_condition', '_value_true', '_value_false',
                 '_shape', '_class_type')
    _attribute_nodes = ('_condition','_value_true','_value_false')
    name = 'where'

    def __new__(cls, condition, x = None, y = None):
        if x is None and y is None:
            return NumpyNonZero(condition)
        elif x is None or y is None:
            raise TypeError("Either both or neither of x and y should be given")
        else:
            return super().__new__(cls)

    def __init__(self, condition, x, y):
        self._condition = condition
        self._value_true = x
        self._value_false = y

        args      = (x, y)
        type_info = NumpyResultType(*args)

        shape = broadcast(x.shape, y.shape)
        shape = broadcast(condition.shape, shape)

        self._shape = process_shape(False, shape)
        rank  = len(shape)
        order = None if rank < 2 else 'C'
        self._class_type = NumpyNDArrayType(process_dtype(type_info.dtype), rank, order)
        super().__init__(condition, x, y)

    @property
    def condition(self):
        """Boolean argument determining which value is returned"""
        return self._condition

    @property
    def value_true(self):
        """Value returned when the condition is evaluated to True."""
        return self._value_true

    @property
    def value_false(self):
        """Value returned when the condition is evaluated to False."""
        return self._value_false

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

#==============================================================================
class NumpyRand(PyccelFunction):
    """
    Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    Represents a call to  numpy.random.random or numpy.random.rand for code generation.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    name = 'rand'

    def __init__(self, *args):
        super().__init__(*args)
        rank  = len(args)
        self._shape = None if rank == 0 else args
        if rank == 0:
            self._class_type = PythonNativeFloat()
        else:
            order = None if rank < 2 else 'C'
            self._class_type = NumpyNDArrayType(NumpyFloat64Type(), rank, order)

#==============================================================================
class NumpyRandint(PyccelFunction):
    """
    Class representing a call to NumPy's randint function.

    Class representing a call to NumPy's randint function.

    Parameters
    ----------
    low : TypedAstNode
        The first argument passed to the function. The smallest possible value for
        the generated number.
    high : TypedAstNode, optional
        The second argument passed to the function. The largest possible value for
        the generated number.
    size : TypedAstNode, optional
        The size of the array that will be generated.
    """
    __slots__ = ('_rand','_low','_high','_shape','_class_type')
    name = 'randint'
    _attribute_nodes = ('_low', '_high')

    def __init__(self, low, high = None, size = None):
        if size is not None and not hasattr(size,'__iter__'):
            size = (size,)

        if high is None:
            high = low
            low  = None

        self._shape   = size
        if size is None:
            self._class_type = PythonNativeInt()
        else:
            rank = len(self.shape)
            order = None if rank < 2 else 'C'
            self._class_type = NumpyNDArrayType(NumpyInt64Type(), rank, order)
        self._rand    = NumpyRand() if size is None else NumpyRand(*size)
        self._low     = low
        self._high    = high
        super().__init__()

    @property
    def rand_expr(self):
        return self._rand

    @property
    def high(self):
        """ return high property of NumpyRandint"""
        return self._high

    @property
    def low(self):
        """ return low property of NumpyRandint"""
        return self._low

#==============================================================================
class NumpyFull(NumpyNewArray):
    """
    Represents a call to `numpy.full` for code generation.

    Represents a call to the NumPy function `full` which creates an array
    of a specified size and shape filled with a specified value.

    Parameters
    ----------
    shape : TypedAstNode
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        For a 1D array this is either a `LiteralInteger` or an expression.
        For a ND array this is a `TypedAstNode` with the class type HomogeneousTupleType.

    fill_value : TypedAstNode
        Fill value.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str, optional
        Datatype for the constructed array.
        If `None` the dtype of the fill value is used.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    """
    __slots__ = ('_fill_value','_shape')
    _attribute_nodes = NumpyNewArray._attribute_nodes + ('_shape',)
    name = 'full'

    def __init__(self, shape, fill_value, dtype=None, order='C'):

        # Convert shape to PythonTuple
        shape = process_shape(False, shape)

        init_dtype = dtype
        # If there is no dtype, extract it from fill_value
        # TODO: must get dtype from an annotated node
        if dtype is None:
            dtype = fill_value.dtype
        dtype = process_dtype(dtype)

        # Cast fill_value to correct type
        if fill_value:
            if fill_value.dtype != dtype:
                cast_func = DtypePrecisionToCastFunction[dtype]
                fill_value = cast_func(fill_value)
        self._shape = shape
        rank  = len(self._shape)
        order = NumpyNewArray._process_order(rank, order)

        class_type = NumpyNDArrayType(dtype, rank, order)

        super().__init__(fill_value, class_type = class_type, init_dtype = init_dtype)

    #--------------------------------------------------------------------------
    @property
    def fill_value(self):
        return self._args[0]

#==============================================================================
class NumpyAutoFill(NumpyFull):
    """ Abstract class for all classes which inherit from NumpyFull but
        the fill_value is implicitly specified
    """
    __slots__ = ()
    def __init__(self, shape, dtype='float', order='C'):
        if not dtype:
            raise TypeError("Data type must be provided")
        super().__init__(shape, Nil(), dtype, order)

#==============================================================================
class NumpyEmpty(NumpyAutoFill):
    """
    Represents a call to numpy.empty for code generation.

    Represents a call to numpy.empty for code generation.

    Parameters
    ----------
    shape : TypedAstNode
        The shape of the array to be created.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str
        The actual dtype passed to the NumPy function.

    order : str, LiteralString
        The order passed to the function.
    """
    __slots__ = ()
    name = 'empty'

    def __init__(self, shape, dtype='float', order='C'):
        super().__init__(shape, dtype, order)

    @property
    def fill_value(self):
        """
        The value with which the array will be filled on initialisation.

        The value with which the array will be filled on initialisation.
        """
        return None

#==============================================================================
class NumpyZeros(NumpyAutoFill):
    """
    Represents a call to numpy.zeros for code generation.

    Represents a call to numpy.zeros for code generation.

    Parameters
    ----------
    shape : TypedAstNode
        The shape passed as argument to the function call.
    dtype : PyccelAstNode | PyccelType | str, default = 'float'
        The datatype specified in the argument of the function call.
    order : str, default='C'
        The order specified in the argument of the function call.
    """
    __slots__ = ()
    name = 'zeros'

    @property
    def fill_value(self):
        """
        The value with which the array will be filled on initialisation.

        The value with which the array will be filled on initialisation.
        """
        return convert_to_literal(0, self.dtype)

#==============================================================================
class NumpyOnes(NumpyAutoFill):
    """
    Represents a call to numpy.ones for code generation.

    Represents a call to numpy.ones for code generation.

    Parameters
    ----------
    shape : TypedAstNode
        The shape passed as argument to the function call.
    dtype : PyccelAstNode | PyccelType | str, default = 'float'
        The datatype specified in the argument of the function call.
    order : str, default='C'
        The order specified in the argument of the function call.
    """
    __slots__ = ()
    name = 'ones'
    @property
    def fill_value(self):
        """
        The value with which the array will be filled on initialisation.

        The value with which the array will be filled on initialisation.
        """
        return convert_to_literal(1, self.dtype)

#==============================================================================
class NumpyFullLike(PyccelFunction):
    """
    Represents a call to numpy.full_like for code generation.

    This wrapper class represents calls to the function numpy.full_like.
    Objects of this class are never present in the Pyccel AST, because the
    class constructor always returns an object of type `NumpyFull`.

    Parameters
    ----------
    a : Variable
        Numpy array which is used as a template.

    fill_value : TypedAstNode
        Scalar value which will be assigned to each entry of the new array.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str, optional
        Type of the data contained in the new array. If None, a.dtype is used.

    order : str, default='K'
        Ordering used for the indices of a multi-dimensional array.

    subok : bool, default=True
        This parameter is currently ignored.

    shape : PythonTuple of TypedAstNode
        Overrides the shape of the array.
        For a 1D array this is either a `LiteralInteger` or an expression.
        For a ND array this is a `TypedAstNode` with the class type HomogeneousTupleType.

    See Also
    --------
    numpy.full_like :
        See documentation of `numpy.full_like`: https://numpy.org/doc/stable/reference/generated/numpy.full_like.html .
    """
    __slots__ = ()
    name = 'full_like'
    def __new__(cls, a, fill_value, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = NumpyResultType(a)
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = NumpyShape(a) if shape is None else shape
        return NumpyFull(shape, fill_value, dtype, order)

#==============================================================================
class NumpyEmptyLike(PyccelFunction):
    """
    Represents a call to numpy.empty_like for code generation.

    This wrapper class represents calls to the function numpy.empty_like.
    Objects of this class are never present in the Pyccel AST, because the
    class constructor always returns an object of type `NumpyEmpty`.

    Parameters
    ----------
    a : Variable
        Numpy array which is used as a template.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str, optional
        Type of the data contained in the new array. If None, a.dtype is used.

    order : str, default='K'
        Ordering used for the indices of a multi-dimensional array.

    subok : bool, default=True
        This parameter is currently ignored.

    shape : PythonTuple of TypedAstNode
        Overrides the shape of the array.
        For a 1D array this is either a `LiteralInteger` or an expression.
        For a ND array this is a `TypedAstNode` with the class type HomogeneousTupleType.

    See Also
    --------
    numpy.empty_like :
        See documentation of `numpy.empty_like`: https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html .
    """
    __slots__ = ()
    name = 'empty_like'

    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = NumpyResultType(a)
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = NumpyShape(a) if shape is None else shape

        return NumpyEmpty(shape, dtype, order)

#==============================================================================
class NumpyOnesLike(PyccelFunction):
    """
    Represents a call to numpy.ones_like for code generation.

    This wrapper class represents calls to the function numpy.ones_like.
    Objects of this class are never present in the Pyccel AST, because the
    class constructor always returns an object of type `NumpyOnes`.

    Parameters
    ----------
    a : Variable
        Numpy array which is used as a template.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str, optional
        Type of the data contained in the new array. If None, a.dtype is used.

    order : str, default='K'
        Ordering used for the indices of a multi-dimensional array.

    subok : bool, default=True
        This parameter is currently ignored.

    shape : PythonTuple of TypedAstNode
        Overrides the shape of the array.
        For a 1D array this is either a `LiteralInteger` or an expression.
        For a ND array this is a `TypedAstNode` with the class type HomogeneousTupleType.

    See Also
    --------
    numpy.ones_like :
        See documentation of `numpy.ones_like`: https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html .
    """
    __slots__ = ()
    name = 'ones_like'
    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = NumpyResultType(a)
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = NumpyShape(a) if shape is None else shape

        return NumpyOnes(shape, dtype, order)

#==============================================================================
class NumpyZerosLike(PyccelFunction):
    """
    Represents a call to numpy.zeros_like for code generation.

    This wrapper class represents calls to the function numpy.zeros_like.
    Objects of this class are never present in the Pyccel AST, because the
    class constructor always returns an object of type `NumpyZeros`.

    Parameters
    ----------
    a : Variable
        Numpy array which is used as a template.

    dtype : PythonType, PyccelFunctionDef, LiteralString, str, optional
        Type of the data contained in the new array. If None, a.dtype is used.

    order : str, default='K'
        Ordering used for the indices of a multi-dimensional array.

    subok : bool, default=True
        This parameter is currently ignored.

    shape : PythonTuple of TypedAstNode
        Overrides the shape of the array.
        For a 1D array this is either a `LiteralInteger` or an expression.
        For a ND array this is a `TypedAstNode` with the class type HomogeneousTupleType.

    See Also
    --------
    numpy.zeros_like :
        See documentation of `numpy.zeros_like`: https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html .
    """
    __slots__ = ()
    name = 'zeros_like'

    def __new__(cls, a, dtype=None, order='K', subok=True, shape=None):

        # NOTE: we ignore 'subok' argument
        if dtype is None:
            dtype = NumpyResultType(a)
        order = a.order if str(order).strip('\'"') in ('K', 'A') else order
        shape = NumpyShape(a) if shape is None else shape

        return NumpyZeros(shape, dtype, order)

#==============================================================================
class NumpyNorm(PyccelFunction):
    """
    Represents call to `numpy.norm`.

    Represents a call to the NumPy function norm.

    Parameters
    ----------
    arg : TypedAstNode
        The first argument passed to the function.
    axis : TypedAstNode, optional
        The second argument passed to the function, indicating the axis along
        which the norm should be calculated.
    """
    __slots__ = ('_shape','_arg','_class_type')
    name = 'norm'

    def __init__(self, arg, axis=None):
        super().__init__(arg, axis)
        arg_dtype = arg.dtype
        if not isinstance(arg_dtype.primitive_type, (PrimitiveFloatingPointType, PrimitiveComplexType)):
            arg = NumpyFloat64(arg)
            dtype = NumpyFloat64Type()
        else:
            dtype = numpy_precision_map[(PrimitiveFloatingPointType(), arg_dtype.precision)]
        self._arg = PythonTuple(arg) if arg.rank == 0 else arg
        if self.axis is not None:
            sh = list(arg.shape)
            del sh[self.axis]
            self._shape = tuple(sh)
            rank = len(self._shape)
            order = None if rank < 2 else arg.order
            self._class_type = NumpyNDArrayType(dtype, rank, order) if rank else dtype
        else:
            self._shape = None
            self._class_type = dtype

    @property
    def arg(self):
        return self._arg

    @property
    def python_arg(self):
        """numpy.norm argument without casting.
        the actual arg property contains casting methods for C/Fortran,
        which is not necessary for a Python code, and the casting makes Python language tests fail.
        """
        return self._args[0]

    @property
    def axis(self):
        """
        Mimic the behavior of axis argument of numpy.norm in python,
        and dim argument of Norm2 in Fortran.
        """
        return self._args[1]

#==============================================================================
# Numpy universal functions
# https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs
#==============================================================================
class NumpyUfuncBase(PyccelFunction):
    """
    Base class for Numpy's universal functions.

    The class from which NumPy's universal functions inherit. All classes which
    inherit from this class operate on their arguments elementally.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ('_shape','_class_type')

    @property
    def is_elemental(self):
        return True

#------------------------------------------------------------------------------
class NumpyUfuncUnary(NumpyUfuncBase):
    """
    Class representing Numpy's universal function with one argument.

    Class representing Numpy's universal function. All classes which
    inherit from this class have one argument and operate on it
    elementally. In other words it should be equivalent to write:
    >>> for i in iterable: NumpyUfuncUnary(i)

    or
    >>> NumpyUfuncUnary(iterable)

    Parameters
    ----------
    x : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()

    def __init__(self, x):
        dtype = self._get_dtype(x)
        self._shape, rank = self._get_shape_rank(x)
        order = self._get_order(x, rank)
        self._class_type = NumpyNDArrayType(dtype, rank, order) if rank else dtype
        super().__init__(x)

    def _get_shape_rank(self, x):
        """
        Get the shape and rank of the result of the function.

        Get the shape and rank of the result of the function.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        shape : tuple[TypedAstNode]
            The shape of the result of the function.
        rank : int
            The rank of the result of the function.
        """
        return x.shape, x.rank

    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype of the result.

        Use the argument to calculate the dtype of the result.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        x_dtype = x.dtype
        if not isinstance(x_dtype.primitive_type, (PrimitiveFloatingPointType, PrimitiveComplexType)):
            return NumpyFloat64Type()
        else:
            return numpy_precision_map[(x_dtype.primitive_type, x_dtype.precision)]

    def _get_order(self, x, rank):
        """
        Get the order of the result of the function.

        Get the order of the result of the function.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        rank : int
            The rank of the result of the function calculated by _get_shape_rank.

        Returns
        -------
        str
            The order of the result of the function.
        """
        return x.order

    @property
    def arg(self):
        """
        The argument passed to the NumPy unary function.

        The argument passed to the NumPy unary function.
        """
        return self._args[0]

#------------------------------------------------------------------------------
class NumpyUfuncBinary(NumpyUfuncBase):
    """
    Class representing Numpy's universal function with two arguments.

    Class representing Numpy's universal function. All classes which
    inherit from this class have two arguments and operate on them
    in lockstep. In other words it should be equivalent to write:
    >>> for i,_ in enumerate(iterable1): NumpyUfuncUnary(iterable1(i), iterable2(i))

    or
    >>> NumpyUfuncUnary(iterable1, iterable2)

    Parameters
    ----------
    x1 : TypedAstNode
        The first argument passed to the function.
    x2 : TypedAstNode
        The second argument passed to the function.
    """
    __slots__ = ()
    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        dtype = self._get_dtype(x1, x2)
        self._shape, rank = self._get_shape_rank(x1, x2)
        order = self._get_order(x1, x2, rank)
        self._class_type = NumpyNDArrayType(dtype, rank, order) if rank else dtype

    def _get_shape_rank(self, x1, x2):
        """
        Get the shape and rank of the result of the function.

        Get the shape and rank of the result of the function.

        Parameters
        ----------
        x1 : TypedAstNode
            The first argument passed to the function.
        x2 : TypedAstNode
            The second argument passed to the function.

        Returns
        -------
        shape : tuple[TypedAstNode]
            The shape of the result of the function.
        rank : int
            The rank of the result of the function.
        """
        shape = broadcast(x1.shape, x2.shape)
        rank  = 0 if shape is None else len(shape)
        return shape, rank

    def _get_dtype(self, x1, x2):
        """
        Use the argument to calculate the dtype of the result.

        Use the argument to calculate the dtype of the result.

        Parameters
        ----------
        x1 : TypedAstNode
            The first argument passed to the function.
        x2 : TypedAstNode
            The second argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        if x1.dtype.primitive_type in (PrimitiveBooleanType(), PrimitiveIntegerType()) or \
           x2.dtype.primitive_type in (PrimitiveBooleanType(), PrimitiveIntegerType()) :
            return NumpyFloat64Type()
        else:
            arg_dtype = x1.dtype + x2.dtype
            return numpy_precision_map[(PrimitiveFloatingPointType(), arg_dtype.precision)]

    def _get_order(self, x1, x2, rank):
        """
        Get the order of the result of the function.

        Get the order of the result of the function.

        Parameters
        ----------
        x1 : TypedAstNode
            The first argument passed to the function.
        x2 : TypedAstNode
            The second argument passed to the function.
        rank : int
            The rank of the result of the function calculated by _get_shape_rank.

        Returns
        -------
        str
            The order of the result of the function.
        """
        if x1.order == x2.order:
            return x1.order
        else:
            return None if rank < 2 else 'C'

#------------------------------------------------------------------------------
# Math operations
#------------------------------------------------------------------------------
#class NumpyAbsolute(NumpyUfuncUnary): __slots__ = ()
class NumpyFabs    (NumpyUfuncUnary):
    """Represent a call to the fabs function in the Numpy library"""
    __slots__ = ()
    name = 'fabs'
class NumpyExp     (NumpyUfuncUnary):
    """Represent a call to the exp function in the Numpy library"""
    __slots__ = ()
    name = 'exp'
class NumpyExpm1   (NumpyUfuncUnary):
    """
    Represent a call to the np.expm1 function in the Numpy library.

    Represent a call to the np.expm1 function in the Numpy library.

    Parameters
    ----------
    x : PyccelAstType
        The argument of the unary function.
    """
    __slots__ = ()
    name = 'expm1'
class NumpyLog     (NumpyUfuncUnary):
    """Represent a call to the log function in the Numpy library"""
    __slots__ = ()
    name = 'log'
class NumpySqrt    (NumpyUfuncUnary):
    """Represent a call to the sqrt function in the Numpy library"""
    __slots__ = ()
    name = 'sqrt'

#------------------------------------------------------------------------------
# Trigonometric functions
#------------------------------------------------------------------------------
class NumpySin    (NumpyUfuncUnary):
    """Represent a call to the sin function in the Numpy library"""
    __slots__ = ()
    name = 'sin'
class NumpyCos    (NumpyUfuncUnary):
    """Represent a call to the cos function in the Numpy library"""
    __slots__ = ()
    name = 'cos'
class NumpyTan    (NumpyUfuncUnary):
    """Represent a call to the tan function in the Numpy library"""
    __slots__ = ()
    name = 'tan'
class NumpyArcsin (NumpyUfuncUnary):
    """Represent a call to the arcsin function in the Numpy library"""
    __slots__ = ()
    name = 'arcsin'
class NumpyArccos (NumpyUfuncUnary):
    """Represent a call to the arccos function in the Numpy library"""
    __slots__ = ()
    name = 'arccos'
class NumpyArctan (NumpyUfuncUnary):
    """Represent a call to the arctan function in the Numpy library"""
    __slots__ = ()
    name = 'arctan'
class NumpyArctan2(NumpyUfuncBinary):
    """Represent a call to the arctan2 function in the Numpy library"""
    __slots__ = ()
    name = 'arctan2'
class NumpyHypot  (NumpyUfuncBinary):
    """Represent a call to the hypot function in the Numpy library"""
    __slots__ = ()
    name = 'hypot'
class NumpySinh   (NumpyUfuncUnary):
    """Represent a call to the sinh function in the Numpy library"""
    __slots__ = ()
    name = 'sinh'
class NumpyCosh   (NumpyUfuncUnary):
    """Represent a call to the cosh function in the Numpy library"""
    __slots__ = ()
    name = 'cosh'
class NumpyTanh   (NumpyUfuncUnary):
    """Represent a call to the tanh function in the Numpy library"""
    __slots__ = ()
    name = 'tanh'
class NumpyArcsinh(NumpyUfuncUnary):
    """Represent a call to the arcsinh function in the Numpy library"""
    __slots__ = ()
    name = 'arcsinh'
class NumpyArccosh(NumpyUfuncUnary):
    """Represent a call to the arccosh function in the Numpy library"""
    __slots__ = ()
    name = 'arccosh'
class NumpyArctanh(NumpyUfuncUnary):
    """Represent a call to the arctanh function in the Numpy library"""
    __slots__ = ()
    name = 'arctanh'
#class NumpyDeg2rad(NumpyUfuncUnary):
#    """Represent a call to the numpydeg2rad function in the Numpy library"""
#    __slots__ = ()
#    name = 'deg2rad'
#class NumpyRad2deg(NumpyUfuncUnary):
#    """Represent a call to the numpyrad2deg function in the Numpy library"""
#     __slots__ = ()
#     name = 'rad2deg'

#==============================================================================
class NumpySign(NumpyUfuncUnary):
    """
    Represent a call to the sign function in the Numpy library.

    Represent a call to the sign function in the Numpy library.

    Parameters
    ----------
    x : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    name = 'sign'
    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype of the result.

        Use the argument to calculate the dtype of the result.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        return process_dtype(x.dtype)

class NumpyAbs(NumpyUfuncUnary):
    """
    Represent a call to the abs function in the Numpy library.

    Represent a call to the abs function in the Numpy library.

    Parameters
    ----------
    x : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    name = 'abs'
    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype of the result.

        Use the argument to calculate the dtype of the result.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        x_dtype = x.dtype
        if isinstance(x_dtype.primitive_type, PrimitiveComplexType):
            dtype = x_dtype.element_type
        else:
            dtype = x_dtype
        return process_dtype(dtype)

class NumpyFloor(NumpyUfuncUnary):
    """
    Represent a call to the floor function in the Numpy library.

    Represent a call to the floor function in the Numpy library.

    Parameters
    ----------
    x : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ()
    name = 'floor'

    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype of the result.

        Use the argument to calculate the dtype of the result.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        if numpy_v2_1:
            return process_dtype(x.dtype)
        else:
            return super()._get_dtype(x)

class NumpyMod(NumpyUfuncBinary):
    """
    Represent a call to the `numpy.mod` function.

    Represent a call to the mod function in the Numpy library.

    Parameters
    ----------
    x1 : TypedAstNode
        Dividend of the operator.
    x2 : TypedAstNode
        Divisor of the operator.
    """
    __slots__ = ()
    name = 'mod'

    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        x1 = NumpyInt(x1) if isinstance(x1.dtype, PythonNativeBool) else x1
        x2 = NumpyInt(x2) if isinstance(x2.dtype, PythonNativeBool) else x2
        self._args = (x1, x2)

    def _get_shape_rank(self, x1, x2):
        """
        Get the shape and rank of the result of the function.

        Get the shape and rank of the result of the function.

        Parameters
        ----------
        x1 : TypedAstNode
            The first argument passed to the function.
        x2 : TypedAstNode
            The second argument passed to the function.

        Returns
        -------
        shape : tuple[TypedAstNode]
            The shape of the result of the function.
        rank : int
            The rank of the result of the function.
        """
        args   = (x1, x2)
        ranks  = [a.rank  for a in args]
        shapes = [a.shape for a in args]

        if all(r == 0 for r in ranks):
            return None, 0
        else:
            if len(args) == 1:
                shape = args[0].shape
            else:
                shape = broadcast(args[0].shape, args[1].shape)

                for a in args[2:]:
                    shape = broadcast(shape, a.shape)

            return shape, len(shape)

    def _get_dtype(self, x1, x2):
        """
        Set the datatype of the object.

        Set the datatype of the object by calculating how the types
        may be promoted.

        Parameters
        ----------
        x1 : TypedAstNode
            The first argument which helps determine the datatype.
        x2 : TypedAstNode
            The second argument which helps determine the datatype.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        if isinstance(x1.dtype, PythonNativeBool) and isinstance(x2.dtype, PythonNativeBool):
            return NumpyInt8Type()
        else:
            arg_class_type = x1.class_type + x2.class_type
            if isinstance(arg_class_type, NumpyNDArrayType):
                arg_dtype = arg_class_type.element_type
            else:
                arg_dtype = arg_class_type
            return process_dtype(arg_dtype)

class NumpyAmin(PyccelFunction):
    """
    Represents a call to  numpy.min for code generation.

    Represents a custom class for handling minimum operations.

    Parameters
    ----------
    arg : array_like
        The input array for which the minimum argument is calculated.
    """
    __slots__ = ('_class_type',)
    name = 'amin'
    _shape = None

    def __init__(self, arg):
        super().__init__(arg)
        self._class_type = arg.dtype

    @property
    def arg(self):
        """
        Get the argument to the min function.
        
        This method retrieves the argument used in the min function.
        """
        return self._args[0]

class NumpyAmax(PyccelFunction):
    """
    Represents a call to  numpy.max for code generation.

    Represents a custom class for handling maximum operations.

    Parameters
    ----------
    arg : array_like
        The input array for which the maximum argument is calculated.
    """
    __slots__ = ('_class_type',)
    name = 'amax'
    _shape = None

    def __init__(self, arg):
        super().__init__(arg)
        self._class_type = arg.dtype

    @property
    def arg(self):
        """
        Get the argument to the max function.

        This method retrieves the argument used in the max function.
        """
        return self._args[0]


    @property
    def is_elemental(self):
        return False

class NumpyTranspose(NumpyUfuncUnary):
    """
    Represents a call to the transpose function from the Numpy library.

    Represents a call to the transpose function from the Numpy library.

    Parameters
    ----------
    x : TypedAstNode
        The array to be transposed.
    *axes : tuple[TypedAstNode]
        The axes along which the user wishes to transpose their array.
    """
    __slots__ = ()
    name = 'transpose'

    def __new__(cls, x, *axes):
        if x.rank<2:
            return x
        else:
            return super().__new__(cls)

    def __init__(self, x, *axes):
        if len(axes)!=0:
            raise NotImplementedError("The axes argument of the transpose function is not yet implemented")
        super().__init__(x)

    @property
    def internal_var(self):
        """ Return the variable being transposed
        """
        return self._args[0]

    def __getitem__(self, *args):
        x = self._args[0]
        rank = x.rank
        # Add empty slices to fully index the object
        if len(args) < rank:
            args = args + tuple([Slice(None, None)]*(rank-len(args)))
        return NumpyTranspose(x.__getitem__(*reversed(args)))

    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype of the result.

        Use the argument to calculate the dtype of the result.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        return process_dtype(x.dtype)

    def _get_shape_rank(self, x):
        """
        Get the shape and rank of the result of the function.

        Get the shape and rank of the result of the function.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        shape : tuple[TypedAstNode]
            The shape of the result of the function.
        rank : int
            The rank of the result of the function.
        """
        shape = tuple(reversed(x.shape))
        rank  = x.rank
        return shape, rank

    def _get_order(self, x, rank):
        """
        Get the order of the result of the function.

        Get the order of the result of the function.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        rank : int
            The rank of the result of the function calculated by _get_shape_rank.

        Returns
        -------
        str
            The order of the result of the function.
        """
        return 'C' if x.order=='F' else 'F'

    @property
    def is_elemental(self):
        return False

class NumpyConjugate(PythonConjugate):
    """
    Represents a call to  numpy.conj for code generation.

    Represents a call to the NumPy function conj or conjugate.
    > a = 1+2j
    > np.conj(a)
    1-2j

    Parameters
    ----------
    arg : TypedAstNode
        The argument passed to the function.
    """
    __slots__ = ('_shape','_class_type')
    name = 'conj'

    def __init__(self, arg):
        super().__init__(arg)
        order = arg.order
        rank  = arg.rank
        self._shape = process_shape(rank == 0, arg.shape)
        self._class_type = NumpyNDArrayType(arg.dtype, rank, order) if rank else arg.dtype

    @property
    def is_elemental(self):
        """ Indicates whether the function should be
        called elementwise for an array argument
        """
        return True

class NumpyNonZeroElement(NumpyNewArray):
    """
    Represents an element of the tuple returned by `NumpyNonZero`.

    Represents an element of the tuple returned by `NumpyNonZero` which
    represents a call to `numpy.nonzero`.

    Parameters
    ----------
    a : TypedAstNode
        The argument which was passed to numpy.nonzero.
    dim : int
        The index of the element in the tuple.
    """
    __slots__ = ('_arr','_dim','_shape')
    _attribute_nodes = ('_arr',)
    name = 'nonzero'

    def __init__(self, a, dim):
        self._arr = a
        self._dim = dim

        self._shape = (NumpyCountNonZero(a),)
        super().__init__(a, class_type = NumpyNDArrayType(NumpyInt64Type(), 1, None))

    @property
    def array(self):
        """ The argument which was passed to numpy.nonzero
        """
        return self._arr

    @property
    def dim(self):
        """ The dimension which the results describe
        """
        return self._dim

class NumpyNonZero(PyccelFunction):
    """
    Class representing a call to the function `numpy.nonzero`.

    Class representing a call to the NumPy function `nonzero` which indicates
    which elements of an array are non-zero.

    Parameters
    ----------
    a : TypedAstNode
        The array argument that was passed to the function.

    Examples
    --------
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
    """
    __slots__ = ('_elements','_arr','_shape')
    _attribute_nodes = ('_elements',)
    name = 'nonzero'
    _class_type = HomogeneousTupleType(NumpyNDArrayType(NumpyInt64Type(), 1, None))

    def __init__(self, a):
        if (a.rank > 1):
            raise NotImplementedError("Non-Zero function is only implemented for 1D arrays")
        self._elements = PythonTuple(*(NumpyNonZeroElement(a, i) for i in range(a.rank)))
        self._arr = a
        self._shape = self._elements.shape
        super().__init__()

    @property
    def array(self):
        """ The array argument
        """
        return self._arr

    @property
    def elements(self):
        """ The elements of the tuple
        """
        return self._elements

    def __iter__(self):
        return self._elements.__iter__()

class NumpyCountNonZero(PyccelFunction):
    """
    Class representing a call to the NumPy function `count_nonzero`.

    Class representing a call to the NumPy function `count_nonzero` which
    counts the number of non-zero elements in an array.

    Parameters
    ----------
    a : TypedAstNode
        An array for which the non-zero elements should be counted.
    axis : int, optional
        The dimension along which the non-zero elements are counted.
    keepdims : LiteralTrue | LiteralFalse
        Indicates if output arrays should have the same number of dimensions
        as arg.
    """
    __slots__ = ('_shape', '_class_type', '_arr',
                '_axis', '_keep_dims')
    _attribute_nodes = ('_arr','_axis')
    name   = 'count_nonzero'

    def __init__(self, a, axis = None, *, keepdims = LiteralFalse()):
        if not isinstance(keepdims, (LiteralTrue, LiteralFalse)):
            errors.report(NON_LITERAL_KEEP_DIMS, symbol=keepdims, severity="fatal")
        if axis is not None and not isinstance(axis, LiteralInteger):
            errors.report(NON_LITERAL_AXIS, symbol=axis, severity="fatal")

        if keepdims.python_value:
            dtype = NumpyInt64Type()
            rank  = a.rank
            order = a.order
            if axis is not None:
                shape = list(a.shape)
                shape[axis.python_value] = LiteralInteger(1)
                self._shape = tuple(shape)
            else:
                self._shape = (LiteralInteger(1),)*rank
            self._class_type = NumpyNDArrayType(dtype, rank, order)
        else:
            if axis is not None:
                dtype = NumpyInt64Type()
                shape = list(a.shape)
                shape.pop(axis.python_value)
                self._shape = tuple(shape)
                rank  = a.rank-1
                order = a.order
                self._class_type = NumpyNDArrayType(dtype, rank, order)
            else:
                self._shape = None
                self._class_type = PythonNativeInt()

        self._arr = a
        self._axis = axis
        self._keep_dims = keepdims

        super().__init__(a)

    @property
    def array(self):
        """ The argument which was passed to numpy.nonzero
        """
        return self._arr

    @property
    def axis(self):
        """ The dimension which the results describe
        """
        return self._axis

    @property
    def keep_dims(self):
        """ Indicates if output arrays should have the same number
        of dimensions as arg
        """
        return self._keep_dims


class NumpySize(PyccelFunction):
    """
    Represent a call to numpy.size in the user code.

    This wrapper class represents a call to the NumPy `size` function, which
    returns the total number of elements in a multidimensional array, or the
    number of elements along a given dimension.

    Objects of this class are never present in the Pyccel AST, because the
    class constructor returns objects of type `PyccelArraySize`, `LiteralInteger`, or
    `PyccelArrayShapeElement`.

    Parameters
    ----------
    a : TypedAstNode
        An array of unknown size.

    axis : TypedAstNode, optional
        The integer dimension along which the size is requested.

    See Also
    --------
    numpy.size :
        See NumPy docs : https://numpy.org/doc/stable/reference/generated/numpy.ma.size.html .
    """
    __slots__ = ()
    name = 'size'

    def __new__(cls, a, axis = None):

        if axis is None:
            return PyccelArraySize(a)

        if isinstance(axis, LiteralInteger) and a.shape is not None:
            return a.shape[axis.python_value]

        return PyccelArrayShapeElement(a, axis)

class NumpyIsNan(NumpyUfuncUnary):
    """ 
    Represents a call to numpy.isnan() function.

    This class encapsulates a call to the Numpy 'isnan' function. It is used to
    check whether the elements of a given array or expression are NaN (Not-a-Number).

    Parameters
    ----------
    x : TypedAstNode
        A Pyccel expression or array to be checked for NaN values.

    See Also
    --------
    numpy.isnan :
        See NumPy docs : https://numpy.org/doc/stable/reference/generated/numpy.isnan.html .
    """
    __slots__ = ()
    name = 'isnan'

    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype and precision of the result.

        Use the argument to calculate the dtype and precision of the result.
        For this class the dtype and precision is a class property.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        return PythonNativeBool()

class NumpyIsInf(NumpyUfuncUnary):
    """ 
    Represents a call to numpy.isinf() function.

    This class represents a call to the Numpy 'isinf' function, which is used
    to determine whether elements in a given array or expression are positive or
    negative infinity.

    Parameters
    ----------
    x : TypedAstNode
        A Pyccel expression or array to be checked for infinity values.

    See Also
    --------
    numpy.isinf :
        See NumPy docs : https://numpy.org/doc/stable/reference/generated/numpy.isinf.html .
    """
    __slots__ = ()
    name = 'isinf'

    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype and precision of the result.

        Use the argument to calculate the dtype and precision of the result.
        For this class the dtype and precision is a class property.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        return PythonNativeBool()

class NumpyIsFinite(NumpyUfuncUnary):
    """ 
    Represents a call to numpy.isfinite() function.

    This class corresponds to a call to the Numpy 'isfinite' function, which is
    used to determine whether elements in a given array or expression are finite
    (neither NaN nor infinity).

    Parameters
    ----------
    x : TypedAstNode
        A Pyccel expression or array to be checked for finiteness.

    See Also
    --------
    numpy.isfinite :
        See NumPy docs : https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html .
    """
    __slots__ = ()
    name = 'isfinite'

    def _get_dtype(self, x):
        """
        Use the argument to calculate the dtype and precision of the result.

        Use the argument to calculate the dtype and precision of the result.
        For this class the dtype and precision is a class property.

        Parameters
        ----------
        x : TypedAstNode
            The argument passed to the function.

        Returns
        -------
        PyccelType
            The dtype of the result of the function.
        """
        return PythonNativeBool()

#==============================================================================
class NumpyNDArray(PyccelFunction):
    """
    A class representing np.ndarray.

    A class representing np.ndarray. np.ndarray is useful for type
    checks. NumpyNDArray is not designed to be instantiated as
    np.ndarray raises a warning when used in code, but as its
    implementation is identical to np.array the __new__ method maps
    to that class so the method is supported.

    Parameters
    ----------
    *args : tuple
        Positional arguments. See NumpyArray.
    **kwargs : dict
        Keyword arguments. See NumpyArray.
    """
    __slots__ = ()
    _dtype = SymbolicType()
    _static_type = NumpyNDArrayType
    name = 'ndarray'

    def __new__(cls, *args, **kwargs):
        return NumpyArray(*args, **kwargs)

#==============================================================================
class NumpyDivide(PyccelDiv):
    """
    Class representing a class to numpy.divide or numpy.true_divide in the user code.

    Class representing a class to numpy.divide or numpy.true_divide in the user code.

    Parameters
    ----------
    x1 : TypedAstNode
        The dividend.
    x2 : TypedAstNode
        The divisor.
    """
    __slots__ = ()
    name = 'divide'
    def __init__(self, x1, x2):
        if x1.rank == 0:
            x1_type = x1.class_type
            x1_np_type = process_dtype(x1_type)
            if x1_type is not x1_np_type:
                x1 = DtypePrecisionToCastFunction[x1_np_type](x1)
        if x2.rank == 0:
            x2_type = x2.class_type
            x2_np_type = process_dtype(x2_type)
            if x2_type is not x2_np_type:
                x2 = DtypePrecisionToCastFunction[x2_np_type](x2)
        super().__init__(x1, x2)

#==============================================================================
DtypePrecisionToCastFunction.update({
    PythonNativeBool()    : NumpyBool,
    NumpyInt8Type()       : NumpyInt8,
    NumpyInt16Type()      : NumpyInt16,
    NumpyInt32Type()      : NumpyInt32,
    NumpyInt64Type()      : NumpyInt64,
    NumpyFloat32Type()    : NumpyFloat32,
    NumpyFloat64Type()    : NumpyFloat64,
    NumpyComplex64Type()  : NumpyComplex64,
    NumpyComplex128Type() : NumpyComplex128,
    })

#==============================================================================
# TODO split numpy_functions into multiple dictionaries following
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.array-creation.html

numpy_linalg_mod = Module('numpy.linalg', (),
    [PyccelFunctionDef('norm', NumpyNorm)])

numpy_random_mod = Module('numpy.random', (),
    [PyccelFunctionDef('rand'   , NumpyRand),
     PyccelFunctionDef('random' , NumpyRand),
     PyccelFunctionDef('randint', NumpyRandint)])

numpy_constants = {
        'pi': Constant(PythonNativeFloat(), 'pi', value=numpy.pi),
    }

numpy_funcs = {
    # ... array creation routines
    'full'      : PyccelFunctionDef('full'      , NumpyFull),
    'empty'     : PyccelFunctionDef('empty'     , NumpyEmpty),
    'zeros'     : PyccelFunctionDef('zeros'     , NumpyZeros),
    'ones'      : PyccelFunctionDef('ones'      , NumpyOnes),
    'full_like' : PyccelFunctionDef('full_like' , NumpyFullLike),
    'empty_like': PyccelFunctionDef('empty_like', NumpyEmptyLike),
    'zeros_like': PyccelFunctionDef('zeros_like', NumpyZerosLike),
    'ones_like' : PyccelFunctionDef('ones_like' , NumpyOnesLike),
    'array'     : PyccelFunctionDef('array'     , NumpyArray),
    'arange'    : PyccelFunctionDef('arange'    , NumpyArange),
    'copy'      : PyccelFunctionDef('copy'      , NumpyArray),
    # ...
    'shape'     : PyccelFunctionDef('shape'     , NumpyShape),
    'size'      : PyccelFunctionDef('size'      , NumpySize),
    'norm'      : PyccelFunctionDef('norm'      , NumpyNorm),
    'int'       : PyccelFunctionDef('int'       , NumpyInt),
    'real'      : PyccelFunctionDef('real'      , NumpyReal),
    'imag'      : PyccelFunctionDef('imag'      , NumpyImag),
    'conj'      : PyccelFunctionDef('conj'      , NumpyConjugate),
    'conjugate' : PyccelFunctionDef('conjugate' , NumpyConjugate),
    'float'     : PyccelFunctionDef('float'     , NumpyFloat),
    'double'    : PyccelFunctionDef('double'    , NumpyFloat64),
    'mod'       : PyccelFunctionDef('mod'       , NumpyMod),
    'float32'   : PyccelFunctionDef('float32'   , NumpyFloat32),
    'float64'   : PyccelFunctionDef('float64'   , NumpyFloat64),
    'bool'      : PyccelFunctionDef('bool'      , NumpyBool),
    'int8'      : PyccelFunctionDef('int8'      , NumpyInt8),
    'int16'     : PyccelFunctionDef('int16'     , NumpyInt16),
    'int32'     : PyccelFunctionDef('int32'     , NumpyInt32),
    'int64'     : PyccelFunctionDef('int64'     , NumpyInt64),
    'complex'   : PyccelFunctionDef('complex'   , NumpyComplex),
    'complex128': PyccelFunctionDef('complex128', NumpyComplex128),
    'complex64' : PyccelFunctionDef('complex64' , NumpyComplex64),
    'matmul'    : PyccelFunctionDef('matmul'    , NumpyMatmul),
    'sum'       : PyccelFunctionDef('sum'       , NumpySum),
    'max'       : PyccelFunctionDef('max'       , NumpyAmax),
    'amax'      : PyccelFunctionDef('amax'      , NumpyAmax),
    'min'       : PyccelFunctionDef('min'       , NumpyAmin),
    'amin'      : PyccelFunctionDef('amin'      , NumpyAmin),
    'prod'      : PyccelFunctionDef('prod'      , NumpyProduct),
    'product'   : PyccelFunctionDef('product'   , NumpyProduct),
    'linspace'  : PyccelFunctionDef('linspace'  , NumpyLinspace),
    'where'     : PyccelFunctionDef('where'     , NumpyWhere),
    'divide'    : PyccelFunctionDef('divide'    , NumpyDivide),
    'true_divide' : PyccelFunctionDef('true_divide', NumpyDivide),
    # ---
    'isnan'     : PyccelFunctionDef('isnan'     , NumpyIsNan),
    'isinf'     : PyccelFunctionDef('isinf'     , NumpyIsInf),
    'isfinite'  : PyccelFunctionDef('isfinite'  , NumpyIsFinite),
    'sign'      : PyccelFunctionDef('sign'      , NumpySign),
    'abs'       : PyccelFunctionDef('abs'       , NumpyAbs),
    'floor'     : PyccelFunctionDef('floor'     , NumpyFloor),
    'absolute'  : PyccelFunctionDef('absolute'  , NumpyAbs),
    'fabs'      : PyccelFunctionDef('fabs'      , NumpyFabs),
    'exp'       : PyccelFunctionDef('exp'       , NumpyExp),
    'expm1'     : PyccelFunctionDef('expm1'     , NumpyExpm1),
    'log'       : PyccelFunctionDef('log'       , NumpyLog),
    'sqrt'      : PyccelFunctionDef('sqrt'      , NumpySqrt),
    # ---
    'sin'       : PyccelFunctionDef('sin'       , NumpySin),
    'cos'       : PyccelFunctionDef('cos'       , NumpyCos),
    'tan'       : PyccelFunctionDef('tan'       , NumpyTan),
    'arcsin'    : PyccelFunctionDef('arcsin'    , NumpyArcsin),
    'arccos'    : PyccelFunctionDef('arccos'    , NumpyArccos),
    'arctan'    : PyccelFunctionDef('arctan'    , NumpyArctan),
    'arctan2'   : PyccelFunctionDef('arctan2'   , NumpyArctan2),
    # 'hypot'     : PyccelFunctionDef('hypot'     , NumpyHypot),
    'sinh'      : PyccelFunctionDef('sinh'      , NumpySinh),
    'cosh'      : PyccelFunctionDef('cosh'      , NumpyCosh),
    'tanh'      : PyccelFunctionDef('tanh'      , NumpyTanh),
    'arcsinh'   : PyccelFunctionDef('arcsinh'   , NumpyArcsinh),
    'arccosh'   : PyccelFunctionDef('arccosh'   , NumpyArccosh),
    'arctanh'   : PyccelFunctionDef('arctanh'   , NumpyArctanh),
    # 'deg2rad'   : PyccelFunctionDef('deg2rad'   , NumpyDeg2rad),
    # 'rad2deg'   : PyccelFunctionDef('rad2deg'   , NumpyRad2deg),
    'transpose' : PyccelFunctionDef('transpose' , NumpyTranspose),
    'nonzero'   : PyccelFunctionDef('nonzero'   , NumpyNonZero),
    'count_nonzero' : PyccelFunctionDef('count_nonzero', NumpyCountNonZero),
    'result_type' : PyccelFunctionDef('result_type', NumpyResultType),
    'dtype'     : PyccelFunctionDef('dtype', NumpyResultType),
    'ndarray'   : PyccelFunctionDef('ndarray', NumpyNDArray),
}

numpy_mod = Module('numpy',
    variables = numpy_constants.values(),
    funcs = numpy_funcs.values(),
    imports = [
        Import('linalg', numpy_linalg_mod),
        Import('random', numpy_random_mod),
        ])

#==============================================================================

numpy_target_swap = {
        numpy_funcs['full_like']  : numpy_funcs['full'],
        numpy_funcs['empty_like'] : numpy_funcs['empty'],
        numpy_funcs['zeros_like'] : numpy_funcs['zeros'],
        numpy_funcs['ones_like']  : numpy_funcs['ones']
    }
