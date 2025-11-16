# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing all classes useful for type annotation.
"""

from pyccel.utilities.stage import PyccelStage

from .basic import PyccelAstNode

from .bitwise_operators import PyccelBitOr

from .core import FunctionDefArgument, FunctionDefResult

from .datatypes import PythonNativeBool, PythonNativeInt, PythonNativeFloat, PythonNativeComplex
from .datatypes import VoidType, GenericType, StringType, PyccelType

from .literals import LiteralString

from .variable import DottedName, AnnotatedPyccelSymbol, IndexedElement

__all__ = (
        'FunctionTypeAnnotation',
        'SyntacticTypeAnnotation',
        'UnionTypeAnnotation',
        'VariableTypeAnnotation',
        'typenames_to_dtypes',
        )

pyccel_stage = PyccelStage()

#==============================================================================

class VariableTypeAnnotation(PyccelAstNode):
    """
    A class which describes a type annotation on a variable.

    A class which stores all information which may be provided in a type annotation
    in order to declare a variable.

    Parameters
    ----------
    class_type : PyccelType
        The requested Python type of the variable.
    """
    __slots__ = ('_class_type',)
    _attribute_nodes = ()

    def __init__(self, class_type : PyccelType):
        self._class_type = class_type

        super().__init__()

    @property
    def class_type(self):
        """
        Get the Python type of the object.

        The Python type of the object. In the case of scalars this is equivalent to
        the datatype. For objects in (homogeneous) containers (e.g. list/ndarray/tuple),
        this is the type of the container.
        """
        return self._class_type

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return self.class_type.rank

    def __hash__(self):
        return hash(self.class_type)

    def __eq__(self, other):
        # Needed for set
        if isinstance(other, VariableTypeAnnotation):
            return self.class_type == other.class_type
        else:
            return False

    def __repr__(self):
        return repr(self._class_type)

#==============================================================================

class FunctionTypeAnnotation(PyccelAstNode):
    """
    A class which describes a type annotation on a function address.

    A class which stores all information necessary to describe the prototype
    of the function being referenced. This includes the type annotations for
    the arguments and the results.

    Parameters
    ----------
    args : list of SyntacticTypeAnnotation | UnionTypeAnnotation
        The type annotations describing the arguments of the function address.
        In the syntactic stage these objects are of type SyntacticTypeAnnotation.
        In the semantic stage these objects are of type UnionTypeAnnotation.

    result : SyntacticTypeAnnotation | UnionTypeAnnotation
        The type annotation describing the result of the function address.
        In the syntactic stage this object is of type SyntacticTypeAnnotation.
        In the semantic stage this object is of type UnionTypeAnnotation.

    is_const : bool, default=True
        True if the function pointer cannot be modified, false otherwise.
    """
    __slots__ = ('_args', '_result', '_is_const')
    _attribute_nodes = ('_args', '_result', '_is_const')

    def __init__(self, args, result, is_const = True):
        if pyccel_stage == 'syntactic':
            self._args = [FunctionDefArgument(AnnotatedPyccelSymbol('_', a), annotation = a) \
                            for i, a in enumerate(args)]
            if result:
                self._result = FunctionDefResult(AnnotatedPyccelSymbol('_', result), annotation = result)
            else:
                self._result = FunctionDefResult(result)
        else:
            self._args = args
            self._result = result

        self._is_const = is_const

        super().__init__()

    @property
    def args(self):
        """
        Get the type annotations describing the arguments of the function address.

        Get the type annotations describing the arguments of the function address.
        In the syntactic stage these objects are of type SyntacticTypeAnnotation.
        In the semantic stage these objects are of type UnionTypeAnnotation.
        """
        return self._args

    @property
    def result(self):
        """
        Get the type annotation describing the result of the function address.

        Get the type annotation describing the result of the function address.
        In the syntactic stage this object is of type SyntacticTypeAnnotation.
        In the semantic stage this object is of type UnionTypeAnnotation.
        """
        return self._result

    def __repr__(self):
        return f'func({repr(self.args)}) -> {repr(self.results)}'

    @property
    def is_const(self):
        """
        Indicates whether the object will remain constant.

        Returns a boolean which is false if the value of the object can be
        modified, and true otherwise.
        """
        return self._is_const

    @is_const.setter
    def is_const(self, val):
        if not isinstance(val, bool):
            raise TypeError("Is const value should be a boolean")
        self._is_const = val

#==============================================================================

class UnionTypeAnnotation(PyccelAstNode):
    """
    A class which holds multiple possible type annotations.

    A class which holds multiple possible type annotations.

    Parameters
    ----------
    *type_annotations : tuple of VariableTypeAnnotation | FunctionTypeAnnotation
        The objects describing the possible type annotations.
    """
    __slots__ = ('_type_annotations',)
    _attribute_nodes = ('_type_annotations',)

    def __init__(self, *type_annotations):
        annots = [ti for t in type_annotations for ti in (t.type_list if isinstance(t, UnionTypeAnnotation) else [t])]
        # Strip out repeats
        self._type_annotations = tuple({a: None for a in annots}.keys())

        super().__init__()

    @property
    def type_list(self):
        """
        Get the list of possible type annotations.

        Get the list of possible type annotations (stored in a tuple).
        """
        return self._type_annotations

    def add_type(self, annot):
        """
        Add an additional type to the type annotations.

        Add an additional type to the type annotations stored in this UnionTypeAnnotation.

        Parameters
        ----------
        annot : VariableTypeAnnotation | FunctionTypeAnnotation
            The object describing the additional type annotation.
        """
        if annot not in self._type_annotations:
            self._type_annotations += (annot,)
            annot.set_current_user_node(self)

    def __len__(self):
        return len(self._type_annotations)

    def __iter__(self):
        return self._type_annotations.__iter__()

    def __str__(self):
        return '|'.join(str(t) for t in self._type_annotations)

#==============================================================================

class SyntacticTypeAnnotation(PyccelAstNode):
    """
    A class describing the type annotation parsed in the syntactic stage.

    A class which holds all the type information parsed from literal string
    annotations in the syntactic stage. Annotations can describe multiple
    different possible types. This function stores lists of the critical
    properties.

    Parameters
    ----------
    dtype : str | IndexedElement | DottedName | LiteralString
        The dtype named in the type annotation.

    order : str | None
        The order requested in the type annotation.
    """
    __slots__ = ('_dtype', '_order')
    _attribute_nodes = ('_dtype',)

    def __new__(cls, dtype = None, order = None):
        if isinstance(dtype, PyccelBitOr):
            return UnionTypeAnnotation(*[SyntacticTypeAnnotation(d) for d in dtype.args])
        else:
            return super().__new__(cls)

    def __init__(self, dtype, order = None):
        if not isinstance(dtype, (str, DottedName, IndexedElement, LiteralString)):
            raise ValueError(f"Syntactic datatypes should be strings not {type(dtype)}")
        if not (order is None or isinstance(order, str)):
            raise ValueError("Order should be a string")
        self._dtype = dtype
        self._order = order
        super().__init__()
        assert self.pyccel_staging == 'syntactic'

    @property
    def dtype(self):
        """
        A list of the dtypes named in the type annotation.

        A list of the dtypes named in the type annotation. These dtypes are
        all strings.
        """
        return self._dtype

    @property
    def order(self):
        """
        A list of the orders requested for each possible annotation.

        A list of strings or None values indicating the order of the generated
        object.
        """
        return self._order

    def __hash__(self):
        return hash((self._dtype, self._order))

    def __eq__(self, o):
        if isinstance(o, SyntacticTypeAnnotation):
            return self.dtype == o.dtype and \
                    self.order == o.order
        else:
            return False

    def __str__(self):
        order_str = f'(order={self.order})' if self.order else ''
        return f'{self.dtype}{order_str}'

#==============================================================================

typenames_to_dtypes = { 'float'   : PythonNativeFloat(),
                        'double'  : PythonNativeFloat(),
                        'complex' : PythonNativeComplex(),
                        'int'     : PythonNativeInt(),
                        'bool'    : PythonNativeBool(),
                        'b1'      : PythonNativeBool(),
                        'void'    : VoidType(),
                        '*'       : GenericType(),
                        'str'     : StringType(),
                        }
