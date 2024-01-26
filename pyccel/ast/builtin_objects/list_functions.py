from ..basic import PyccelAstNode
from pyccel.utilities.stage import PyccelStage
from ..datatypes import(NativeInteger, NativeBool, NativeFloat,
                        NativeComplex, NativeString, str_dtype,
                        NativeGeneric, default_precision)
from ..internals import max_precision, PyccelInternalFunction
from ..literals import LiteralFalse, LiteralInteger
pyccel_stage = PyccelStage()

class PythonList(PyccelAstNode):
    """ Represents a call to Python's native list() function.
    """
    __slots__ = ('_args', '_is_homogeneous',
            '_dtype', '_precision', '_rank', '_shape', '_order')
    _iterable = True
    _attribute_nodes = ('_args',)

    def __init__(self, *args):

        # TODO: Need to include empty lists. Now, an Inhomogeneous list error is raised

        self._args = args
        super().__init__()
        if pyccel_stage == 'syntactic':
            return
        if len(args) == 0:
            self._dtype = NativeGeneric()
            self._precision = 0
            self._rank = 0
            self._shape = None
            self._order = None
            self._is_homogeneous = True
        else:
            a0_precision = args[0].precision
            a0_dtype     = args[0].dtype
            a0_shape     = args[0].shape
            a0_order     = args[0].order
            a0_rank      = args[0].rank
            is_homogeneous  = not isinstance(args[0], NativeGeneric) and \
                            all(not isinstance(a, NativeGeneric) and \
                                a.dtype     == a0_dtype and \
                                a.precision == a0_precision and \
                                a.rank      == a0_rank and \
                                a.shape     == a0_shape and \
                                a.order     == a0_order \
                                for a in args[1:])
            self._is_homogeneous = is_homogeneous
            if is_homogeneous:
                strts    = [a for a in self._args if a.dtype == NativeString()]
                integers = [a for a in self._args if a.dtype == NativeInteger()]
                floats   = [a for a in self._args if a.dtype == NativeFloat()]
                cmplxs   = [a for a in self._args if a.dtype == NativeComplex()]
                bools    = [a for a in self._args if a.dtype == NativeBool()]

                if strts:
                    self._dtype = NativeString()
                    self._precision = 0
                elif integers:
                    self._dtype = NativeInteger()
                    self._precision = max_precision(integers)
                elif floats:
                    self._dtype = NativeFloat()
                    self._precision = max_precision(floats)
                elif cmplxs:
                    self._dtype = NativeComplex()
                    self._precision = max_precision(cmplxs)
                elif bools:
                    self._dtype = NativeBool()
                    self._precision = max_precision(bools)
                else:
                    raise TypeError('Cannot determine the type of {}'.format(self))

                self._rank = self._args[0].rank + 1
                self._order = None if self._rank < 2 else 'C'
                self._shape = (LiteralInteger(len(self._args)),) + (self._args[0].shape or ())

            else:
                raise TypeError('Inhomogeneous lists are not supported.')

    def __getitem__(self, i):
        #TODO: ensure i is integer
        return self._args[i]

    def __iter__(self):
        return self._args.__iter__()

    def __len__(self):
        return len(self._args)

    def __str__(self):
        return '[{}]'.format(', '.join(str(a) for a in self._args))

    def __repr__(self):
        return 'PythonList({})'.format(', '.join(str(a) for a in self._args))

    def __add__(self, other):
        return PythonList(*(self.args + other.args))

    @property
    def args(self):
        """ Arguments of the list
        """
        return self._args

    @property
    def is_homogeneous(self):
        """ Indicates if the list is homogeneous or not
        """
        return self._is_homogeneous

class PythonListMethod(PyccelInternalFunction):
    __slots__ = ('_list', '_args')
    _dtype = NativeInteger()
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
