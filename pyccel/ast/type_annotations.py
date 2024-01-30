# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing all classes useful for type annotation.
"""

from pyccel.utilities.stage import PyccelStage

from .basic import PyccelAstNode

from .core import FunctionDefArgument

from .variable import DottedName, AnnotatedPyccelSymbol, IndexedElement

__all__ = (
        'FunctionTypeAnnotation',
        'SyntacticTypeAnnotation',
        'VariableTypeAnnotation',
        'UnionTypeAnnotation',
        )

pyccel_stage = PyccelStage()

class VariableTypeAnnotation(PyccelAstNode):
    """
    A class which describes a type annotation on a variable.

    A class which stores all information which may be provided in a type annotation
    in order to declare a variable.

    Parameters
    ----------
    datatype : DataType
        The requested internal data type.

    class_type : DataType
        The Python type of the variable. In the case of scalars this is equivalent to
        the datatype. For objects in (homogeneous) containers (e.g. list/ndarray/tuple),
        this is the type of the container.

    precision : int
        The precision of the internal datatype.

    rank : int
        The rank of the variable.

    order : str
        The order of the variable.

    is_const : bool, default=False
        True if the variable cannot be modified, false otherwise.
    """
    __slots__ = ('_datatype', '_class_type', '_precision', '_rank',
                 '_order', '_is_const')
    _attribute_nodes = ()
    def __init__(self, datatype : 'DataType', class_type : 'DataType', precision : int = -1,
            rank : int = 0, order : str = None, is_const : bool = False):
        self._datatype = datatype
        self._class_type = class_type
        self._precision = precision
        self._rank = rank
        self._order = order
        self._is_const = is_const

        super().__init__()

    @property
    def datatype(self):
        """
        Get the basic datatype of the object.

        Get the basic datatype of the object. For objects with rank>0 this is the
        type of one of the elements of the object.
        """
        return self._datatype

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
    def precision(self):
        """
        Get the precision of the object.

        Get the precision of the object. For objects with rank>0 this is the
        precision of one of the elements of the object.
        """
        return self._precision

    @property
    def rank(self):
        """
        Get the rank of the object.

        Get the rank of the object that should be created. The rank indicates the
        number of dimensions.
        """
        return self._rank

    @property
    def order(self):
        """
        Get the order of the object.

        Get the order in which the memory will be laid out in the object. For objects
        with rank > 1 this is either 'C' or 'F'. 
        """
        return self._order

    @order.setter
    def order(self, order):
        if order not in ('C', 'F', None):
            raise ValueError("Order must be C, F, or None")
        self._order = order

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

    def __hash__(self):
        return hash((self.datatype, self.class_type, self.precision, self.rank, self.order))

    def __eq__(self, other):
        # Needed for set
        if isinstance(other, VariableTypeAnnotation):
            return self.datatype == other.datatype and \
                   self.class_type == other.class_type and \
                   self.precision == other.precision and \
                   self.rank == other.rank and \
                   self.order == other.order
        else:
            return False

    def __repr__(self):
        return f"{self._datatype}{self._precision}[{self._rank}]({self._order})"

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

    results : list of SyntacticTypeAnnotation | UnionTypeAnnotation
        The type annotations describing the results of the function address.
        In the syntactic stage these objects are of type SyntacticTypeAnnotation.
        In the semantic stage these objects are of type UnionTypeAnnotation.

    is_const : bool, default=True
        True if the function pointer cannot be modified, false otherwise.
    """
    __slots__ = ('_args', '_results', '_is_const')
    _attribute_nodes = ('_args', '_results', '_is_const')

    def __init__(self, args, results, is_const = True):
        if pyccel_stage == 'syntactic':
            self._args = [FunctionDefArgument(AnnotatedPyccelSymbol('_', a), annotation = a) \
                            for i, a in enumerate(args)]
            self._results = [FunctionDefArgument(AnnotatedPyccelSymbol('_', r), annotation = r) \
                            for i, r in enumerate(results)]
        else:
            self._args = args
            self._results = results

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
    def results(self):
        """
        Get the type annotations describing the results of the function address.

        Get the type annotations describing the results of the function address.
        In the syntactic stage these objects are of type SyntacticTypeAnnotation.
        In the semantic stage these objects are of type UnionTypeAnnotation.
        """
        return self._results

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
        self._type_annotations = tuple(set(annots))

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

class SyntacticTypeAnnotation(PyccelAstNode):
    """
    A class describing the type annotation parsed in the syntactic stage.

    A class which holds all the type information parsed from literal string
    annotations in the syntactic stage. Annotations can describe multiple
    different possible types. This function stores lists of the critical
    properties.

    Parameters
    ----------
    dtype : PyccelSymbol | IndexedElement | DottedName
        The dtype named in the type annotation.

    order : str | None
        The order requested in the type annotation.
    """
    __slots__ = ('_dtype', '_order')
    _attribute_nodes = ()
    def __init__(self, dtype, order = None):
        if not isinstance(dtype, (str, DottedName, IndexedElement)):
            raise ValueError("Syntactic datatypes should be strings")
        if not (order is None or isinstance(order, str)):
            raise ValueError("Order should be a string")
        self._dtype = dtype
        self._order = order
        super().__init__()

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
