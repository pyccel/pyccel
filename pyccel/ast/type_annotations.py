# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing all classes useful for type annotation.
"""

from .basic import PyccelAstNode

__all__ = (
        'SyntacticTypeAnnotation',
        'VariableTypeAnnotation',
        'UnionTypeAnnotation',
        )

class VariableTypeAnnotation(PyccelAstNode):
    """
    A class which describes a type annotation.

    A class which stores all information which may be provided in a type annotation
    in order to declare a variable.

    Parameters
    ----------
    datatype : DataType
        The requested internal data type.

    cls_base : ClassDef
        The description of the class describing the variable.

    precision : int
        The precision of the internal datatype.

    rank : int
        The rank of the variable.

    order : str
        The order of the variable.

    is_const : bool, default=False
        True if the variable cannot be modified, false otherwise.
    """
    __slots__ = ('_datatype', '_cls_base', '_precision', '_rank',
                 '_order', '_is_const')
    _attribute_nodes = ()
    def __init__(self, datatype : 'DataType', cls_base : 'ClassDef', precision : int = -1,
            rank : int = 0, order : str = None, is_const : bool = False):
        self._datatype = datatype
        self._cls_base = cls_base
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
    def cls_base(self):
        """
        Get the class description of the object.

        Get the class def object which describes how the user can interact with the
        future variable.
        """
        return self._cls_base

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

    @property
    def is_const(self):
        """
        Indicates whether the object will remain constant.

        Returns a boolean which is false if the value of the object can be
        modified, and true otherwise.
        """
        return self._is_const

    def __hash__(self):
        return hash((self.datatype, self.cls_base, self.precision, self.rank, self.order))

    def __repr__(self):
        return f"{self._datatype}{self._precision}[{self._rank}]({self._order})"

class UnionTypeAnnotation(PyccelAstNode):
    """
    A class which holds multiple possible type annotations.

    A class which holds multiple possible type annotations.

    Parameters
    ----------
    *type_annotations : tuple of VariableTypeAnnotation
        The VariableTypeAnnotation objects describing the possible type annotations.
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

class SyntacticTypeAnnotation(PyccelAstNode):
    """
    A class describing the type annotation parsed in the syntactic stage.

    A class which holds all the type information parsed from literal string
    annotations in the syntactic stage. Annotations can describe multiple
    different possible types. This function stores lists of the critical
    properties.

    Parameters
    ----------
    dtypes : list of str
        The dtypes named in the type annotation.

    ranks : list of int
        The number of ranks requested for each possible annotation.

    orders : list of str or None
        The orders requested in the type annotation.

    is_const : bool, optional
        The constness as specified in the type annotation.
        If the constness is unknown then its value will be fixed in the
        semantic stage.
    """
    __slots__ = ('_dtypes', '_ranks', '_orders', '_is_const')
    _attribute_nodes = ()
    def __init__(self, dtypes, ranks, orders, is_const = None):
        if any(not isinstance(d, str) for d in dtypes):
            raise ValueError("Syntactic datatypes should be strings")
        if any(not isinstance(r, int) for r in ranks):
            raise ValueError("Ranks should have integer values")
        if not all(o is None or isinstance(o, str) for o in orders):
            raise ValueError("Orders should be strings")
        if not (isinstance(is_const, bool) or is_const is None):
            raise ValueError("Is const should be a boolean")
        self._dtypes = dtypes
        self._ranks = ranks
        self._orders = [o if o != '' else None for o in orders]
        self._is_const = is_const
        super().__init__()

    @property
    def dtypes(self):
        """
        A list of the dtypes named in the type annotation.

        A list of the dtypes named in the type annotation. These dtypes are
        all strings.
        """
        return self._dtypes

    @property
    def ranks(self):
        """
        A list of the ranks requested for each possible annotation.

        A list of integers indicating the number of ranks of the generated
        object.
        """
        return self._ranks

    @property
    def orders(self):
        """
        A list of the orders requested for each possible annotation.

        A list of strings or None values indicating the order of the generated
        object.
        """
        return self._orders

    @property
    def is_const(self):
        """
        Indicates whether the variable should remain constant.

        Returns a boolean which is false if the value of the variable can be
        modified, and false otherwise.
        """
        return self._is_const

    @staticmethod
    def build_from_textx(annotation):
        """
        Build a SyntacticTypeAnnotation from a textx annotation.

        Build a SyntacticTypeAnnotation from a textx annotation. This function should
        be moved to the SyntacticParser once headers are deprecated.

        Parameters
        ----------
        annotation : object
            An object created by textx. Once headers are deprecated this object will
            have type parser.syntax.basic.BasicStmt.

        Returns
        -------
        SyntacticTypeAnnotation
            A new SyntacticTypeAnnotation describing the parsed type.

        Raises
        ------
        TypeError
            Raised if the type of the argument is not handled.
        """
        if hasattr(annotation, 'dtype'):
            is_const = None
            dtype_names = [annotation.dtype]
            ranks = [len(getattr(annotation.trailer, 'args', ()))]
            orders = [getattr(annotation.trailer, 'order', None)]
            return SyntacticTypeAnnotation(dtype_names, ranks, orders, is_const)
        else:
            raise TypeError("Unexpected type")

