# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.parser.syntax.basic import BasicStmt

from .basic import Basic

from .core import FunctionDefArgument, FunctionDefResult

from .internals import PyccelSymbol

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
            self._args = [FunctionDefArgument(PyccelSymbol('_'), annotation = a) \
                            for i, a in enumerate(args)]
            self._results = [FunctionDefArgument(PyccelSymbol('_'), annotation = r) \
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
        if any(not isinstance(t, (TypeAnnotation, UnionTypeAnnotation, FunctionTypeAnnotation)) for t in type_annotations):
            raise TypeError("Type annotations should have type TypeAnnotation")

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

