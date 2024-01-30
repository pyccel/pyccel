from ..basic import PyccelAstNode, TypedAstNode
from pyccel.utilities.stage import PyccelStage
from ..datatypes import(NativeInteger, NativeBool, NativeFloat,
                        NativeComplex, NativeString, str_dtype, NativeHomogeneousList,
                        NativeGeneric, default_precision, )
from ..internals import max_precision, PyccelInternalFunction,get_final_precision

from ..literals import LiteralFalse, LiteralInteger
pyccel_stage = PyccelStage()

# class PythonList(TypedAstNode):
#     """
#     Class representing a call to Python's `[,]` function.

#     Class representing a call to Python's `[,]` function which generates
#     a literal Python list.

#     Parameters
#     ----------
#     *args : tuple of TypedAstNodes
#         The arguments passed to the operator.

#     See Also
#     --------
#     FunctionalFor
#         The `[]` function when it describes a comprehension.
#     """
#     __slots__ = ('_args','_dtype','_precision','_rank','_shape','_order')
#     _attribute_nodes = ('_args',)
#     _class_type = NativeHomogeneousList()

#     def __init__(self, *args):
#         self._args = args
#         super().__init__()
#         if pyccel_stage == 'syntactic':
#             return
#         elif len(args) == 0:
#             self._dtype = NativeGeneric()
#             self._precision = 0
#             self._rank  = 0
#             self._shape = None
#             self._order = None
#             return
#         arg0 = args[0]
#         precision = get_final_precision(arg0)
#         is_homogeneous = arg0.dtype is not NativeGeneric() and \
#                          all(a.dtype is not NativeGeneric() and \
#                              arg0.dtype == a.dtype and \
#                              precision == get_final_precision(a) and \
#                              arg0.rank  == a.rank  and \
#                              arg0.order == a.order for a in args[1:])
#         if is_homogeneous:
#             self._dtype = arg0.dtype
#             self._precision = arg0.precision

#             inner_shape = [() if a.rank == 0 else a.shape for a in args]
#             self._rank = max(a.rank for a in args) + 1
#             self._shape = (LiteralInteger(len(args)), ) + inner_shape[0]
#             self._rank  = len(self._shape)

#         else:
#             raise TypeError("Can't create an inhomogeneous list")

#         self._order = None if self._rank < 2 else 'C'

#     def __iter__(self):
#         return self._args.__iter__()

#     def __str__(self):
#         args = ', '.join(str(a) for a in self)
#         return f'({args})'

#     def __repr__(self):
#         args = ', '.join(str(a) for a in self)
#         return f'PythonList({args})'

#     @property
#     def args(self):
#         """
#         Arguments of the list.

#         The arguments that were used to initialise the list.
#         """
#         return self._args

#     @property
#     def is_homogeneous(self):
#         """
#         Indicates whether the list is homogeneous or inhomogeneous.

#         Indicates whether all elements of the list have the same dtype, precision,
#         rank, etc (homogenous) or if these values can vary (inhomogeneous). Lists
#         are always homogeneous.
#         """
#         return True

class PythonListMethod(PyccelInternalFunction):
    __slots__ = ('_list', '_args')
    _dtype = NativeInteger()
    _class_type = NativeHomogeneousList()

    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    def __init__(self, *args):
        super().__init__(*args)
        self._list = args[0]
        self._args = args[1:]

    @property
    def list(self):
        return self._list

    @list.setter
    def list(self, other):
        self._list = other

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = args

    def __repr__(self):
        args = ', '.join([str(arg) for arg in self.args])
        return f"{self.list}.{self.name}({args})"

class PythonListAppend(PythonListMethod):
    name = 'append'
    def __init__(self, *args):
        super().__init__(*args)

class PythonListSort(PythonListMethod):
    name = 'sort'

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._reverse = LiteralFalse()
        for arg in self.args:
            if arg.keyword == 'reverse':
                self._reverse = arg.value

    @property
    def reverse(self):
        return self._reverse

    @reverse.setter
    def reverse(self, value):
        self._reverse = value

class PythonListClear(PythonListMethod):
    name = 'clear'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListExtend(PythonListMethod):
    name = 'extend'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListInsert(PythonListMethod):
    name = 'insert'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListCount(PythonListMethod):
    name = 'count'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListPop(PythonListMethod):
    name = 'pop'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListReverse(PythonListMethod):
    name = 'reverse'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListIndex(PythonListMethod):
    name = 'index'

    def __init__(self, *args):
        super().__init__(*args)

class PythonListRemove(PythonListMethod):
    name = 'remove'

    def __init__(self, *args):
        super().__init__(*args)
