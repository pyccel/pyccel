# coding: utf-8

#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

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

    @parameter
    def datatype(self):
        return self._datatype

    @parameter
    def cls_base(self):
        return self._cls_base

    @parameter
    def precision(self):
        return self._precision

    @parameter
    def rank(self):
        return self._rank

    @parameter
    def order(self):
        return self._order

    @parameter
    def is_const(self):
        return self._is_const

class UnionTypeAnnotation(Basic):
    __slots__ = ('_type_annotations',)
    _attribute_nodes = ('_type_annotations',)
    def __new__(cls, *type_annotations):
        if len(type_annotations) == 1:
            return type_annotations[0]
        else:
            super().__new__(cls)

    def __init__(self, *type_annotations):
        self._type_annotations = type_annotations

        if any(not isinstance(t, TypeAnnotation) for t in type_annotations):
            raise TypeError("Type annotations should have type TypeAnnotation")

        super().__init__()
