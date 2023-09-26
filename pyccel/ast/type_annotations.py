# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing all classes useful for type annotation.
"""

from pyccel.utilities.stage import PyccelStage

from .basic import Basic

from .core import FunctionDefArgument

from .internals import AnnotatedPyccelSymbol

__all__ = (
        'FunctionTypeAnnotation',
        'SyntacticTypeAnnotation',
        'TypeAnnotation',
        'UnionTypeAnnotation',
        )

pyccel_stage = PyccelStage()

class TypeAnnotation(Basic):
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
        return self._datatype

    @property
    def cls_base(self):
        return self._cls_base

    @property
    def precision(self):
        return self._precision

    @property
    def rank(self):
        return self._rank

    @property
    def order(self):
        return self._order

    @property
    def is_const(self):
        return self._is_const

    def __eq__(self, other):
        if isinstance(other, TypeAnnotation):
            return self.datatype == other.datatype and \
                   self.cls_base == other.cls_base and \
                   self.precision == other.precision and \
                   self.rank == other.rank and \
                   self.order == other.order
        else:
            return False

    def __hash__(self):
        return hash((self.datatype, self.cls_base, self.precision, self.rank, self.order))

    def __repr__(self):
        return f"{self._datatype}{self._precision}[{self._rank}]({self._order})"

class FunctionTypeAnnotation(Basic):
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
        return self._args

    @property
    def results(self):
        return self._results

    def __repr__(self):
        return f'func({repr(self.args)}) -> {repr(self.results)}'

class UnionTypeAnnotation(Basic):
    __slots__ = ('_type_annotations',)
    _attribute_nodes = ('_type_annotations',)

    def __init__(self, *type_annotations):
        annots = [ti for t in type_annotations for ti in (t.type_list if isinstance(t, UnionTypeAnnotation) else [t])]
        self._type_annotations = tuple(set(annots))

        super().__init__()

    @property
    def type_list(self):
        return self._type_annotations

    def add_type(self, annot):
        self._type_annotations += (annot,)
        annot.set_current_user_node(self)

class SyntacticTypeAnnotation(Basic):
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

    orders : list of str
        The orders requested in the type annotation.

    is_const : bool
        The constness as specified in the type annotation.
    """
    _attribute_nodes = ()
    def __init__(self, dtypes, ranks, orders, is_const):
        self._dtypes = list(dtypes)
        self._ranks = list(ranks)
        self._orders = [o if o != '' else None for o in orders]
        self._is_const = is_const
        super().__init__()

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def ranks(self):
        return self._ranks

    @property
    def orders(self):
        return self._orders

    @property
    def is_const(self):
        return self._is_const

    @staticmethod
    def build_from_textx(annotation):
        if isinstance(annotation, (list, tuple)):
            return tuple(SyntacticTypeAnnotation.build_from_textx(a) for a in annotation)
        elif hasattr(annotation, 'const'):
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
            is_const = None
            dtype_names = [annotation.dtype]
            ranks = [len(getattr(annotation.trailer, 'args', ()))]
            orders = [getattr(annotation.trailer, 'order', None)]
            return SyntacticTypeAnnotation(dtype_names, ranks, orders, is_const)
        elif hasattr(annotation, 'results'):
            args = [SyntacticTypeAnnotation.build_from_textx(a) for a in annotation.decs]
            results = [SyntacticTypeAnnotation.build_from_textx(r) for r in annotation.results]
            return FunctionTypeAnnotation(args, results)
        else:
            raise TypeError("Unexpected type")

