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

from .internals import AnnotatedPyccelSymbol

from .variable import DottedName

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

    def __eq__(self, other):
        # Needed for set
        if isinstance(other, VariableTypeAnnotation):
            return self.datatype == other.datatype and \
                   self.cls_base == other.cls_base and \
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
    """
    __slots__ = ('_args', '_results',)
    _attribute_nodes = ('_args', '_results')

    def __init__(self, args, results):
        if pyccel_stage == 'syntactic':
            self._args = [FunctionDefArgument(AnnotatedPyccelSymbol('_', a), annotation = a) \
                            for i, a in enumerate(args)]
            self._results = [FunctionDefArgument(AnnotatedPyccelSymbol('_', r), annotation = r) \
                            for i, r in enumerate(results)]
        else:
            self._args = args
            self._results = results

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
        if any(not isinstance(d, (str, DottedName)) for d in dtypes):
            raise ValueError("Syntactic datatypes should be strings")
        if any(not isinstance(r, int) for r in ranks):
            raise ValueError("Ranks should have integer values")
        if not all(o is None or isinstance(o, str) for o in orders):
            raise ValueError("Orders should be strings")
        if not (isinstance(is_const, bool) or is_const is None):
            raise ValueError("Is const should be a boolean")
        self._dtypes = tuple(dtypes)
        self._ranks = tuple(ranks)
        self._orders = tuple(o if o != '' else None for o in orders)
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

    def __hash__(self):
        return hash((self._dtypes, self._ranks, self._orders))

    def __eq__(self, o):
        if isinstance(o, SyntacticTypeAnnotation):
            return self.dtypes == o.dtypes and \
                    self.ranks == o.ranks and \
                    self.orders == o.orders
        else:
            return False

    @staticmethod
    def build_from_textx(annotation):
        """
        Build a SyntacticTypeAnnotation from a textx annotation.

        Build a SyntacticTypeAnnotation from a textx annotation. This function should
        be moved to the SyntacticParser once headers are deprecated. When that is
        done there should only be 1 textx object handling types so the if conditions
        can be changed to use isinstance.

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
        if isinstance(annotation, (list, tuple)):
            return tuple(SyntacticTypeAnnotation.build_from_textx(a) for a in annotation)
        elif hasattr(annotation, 'const'):
            # Handle UnionTypeStmt
            is_const = annotation.const
            dtypes = [SyntacticTypeAnnotation.build_from_textx(a) for a in annotation.dtypes]
            if any(isinstance(d, FunctionTypeAnnotation) for d in dtypes):
                if any(not isinstance(d, FunctionTypeAnnotation) for d in dtypes):
                    raise TypeError("Can't mix function address with basic types")
                return UnionTypeAnnotation(*dtypes)
            else:
                dtype_names = [n for d in dtypes for n in d.dtypes]
                ranks = [r for d in dtypes for r in d.ranks]
                orders = [o for d in dtypes for o in d.orders]
                return SyntacticTypeAnnotation(dtype_names, ranks, orders, is_const)
        elif hasattr(annotation, 'dtype'):
            # Handle VariableType
            is_const = None
            dtype_names = [annotation.dtype]
            ranks = [len(getattr(annotation.trailer, 'args', ()))]
            orders = [getattr(annotation.trailer, 'order', None)]
            return SyntacticTypeAnnotation(dtype_names, ranks, orders, is_const)
        elif hasattr(annotation, 'results'):
            # Handle FuncType
            args = [SyntacticTypeAnnotation.build_from_textx(a) for a in annotation.decs]
            results = [SyntacticTypeAnnotation.build_from_textx(r) for r in annotation.results]
            return FunctionTypeAnnotation(args, results)
        else:
            raise TypeError("Unexpected type")

