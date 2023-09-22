# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.parser.syntax.basic import BasicStmt

from .basic import Basic

from pyccel.utilities.stage import PyccelStage

pyccel_stage = PyccelStage()

class TypeAnnotation(Basic):
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

        if rank > 1:
            assert order is not None

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

class UnionTypeAnnotation(Basic):
    __slots__ = ('_type_annotations',)
    _attribute_nodes = ('_type_annotations',)

    def __init__(self, *type_annotations):
        if any(not isinstance(t, (TypeAnnotation, UnionTypeAnnotation)) for t in type_annotations):
            raise TypeError("Type annotations should have type TypeAnnotation")

        annots = [ti for t in type_annotations for ti in ([t] if isinstance(t, TypeAnnotation) else t.type_list)]
        self._type_annotations = tuple(set(annots))

        super().__init__()

    @property
    def type_list(self):
        return self._type_annotations

    def add_type(self, annot):
        self._type_annotations += (annot,)
        annot.set_current_user_node(self)

class SyntacticTypeAnnotation(Basic):
    _attribute_nodes = ()
    def __init__(self, dtypes, ranks, orders, is_const):
        self._dtypes = list(dtypes)
        self._ranks = list(dtypes)
        self._orders = list(dtypes)
        self._is_const = is_const
        super().__init__()

    @staticmethod
    def build_from_textx(annotation):
        if isinstance(annotation, (list, tuple)):
            return tuple(SyntacticTypeAnnotation.build_from_textx(a) for a in annotation)
        elif hasattr(annotation, 'const'):
            is_const = annotation.const
            dtypes = annotation.dtypes
            dtype_names = [d.dtype for d in dtypes]
            ranks = [len(getattr(d.trailer, 'args', ())) for d in dtypes]
            orders = [getattr(d.trailer, 'order', None) for d in dtypes]
            return SyntacticTypeAnnotation(dtype_names, ranks, orders, is_const)
        elif hasattr(annotation, 'dtype'):
            is_const = None
            dtype_names = [annotation.dtype]
            ranks = [len(getattr(annotation.trailer, 'args', ()))]
            orders = [getattr(annotation.trailer, 'order', None)]
            return SyntacticTypeAnnotation(dtype_names, ranks, orders, is_const)
        else:
            raise TypeError("Unexpected type")
