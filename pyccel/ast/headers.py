# coding: utf-8

# TODO must use Header.__new__ rather than Basic.__new__

from sympy.utilities.iterables import iterable
from sympy.core import Symbol
from sympy import sympify

from .core import Basic
from .core import Variable
from .core import FunctionDef
from .core import ClassDef
from .datatypes import datatype, DataTypeFactory, UnionType
from .macros import Macro, MacroSymbol, MacroShape, construct_macro

class Header(Basic):
    pass

class MetaVariable(Header):
    """Represents the MetaVariable."""

    def __new__(cls, name, value):
        if not isinstance(name, str):
            raise TypeError('name must be of type str')

        # TODO check value

        return Basic.__new__(cls, name, value)

    @property
    def name(self):
        return self._args[0]

    @property
    def value(self):
        return self._args[1]

# TODO rename dtypes to arguments
class VariableHeader(Header):
    """Represents a variable header in the code.

    name: str
        variable name

    dtypes: dict
        a dictionary for typing

    Examples

    """

    # TODO dtypes should be a dictionary (useful in syntax)
    def __new__(cls, name, dtypes):
        if not(isinstance(dtypes, dict)):
            raise TypeError("Expecting dtypes to be a dict.")

        return Basic.__new__(cls, name, dtypes)

    @property
    def name(self):
        return self._args[0]

    @property
    def dtypes(self):
        return self._args[1]

# TODO rename dtypes to arguments

    def __getnewargs__(self):
        """used for Pickling self."""
        # TODO improve after renaming the args property
        args = (self._args[0],)
        return args

class FunctionHeader(Header):
    """Represents function/subroutine header in the code.

    func: str
        function/subroutine name

    dtypes: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr, allocatable)

    results: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr, allocatable)

    kind: str
        'function' or 'procedure'. default value: 'function'

    is_static: bool
        True if we want to pass arrays in f2py mode. every argument of type
        array will be preceeded by its shape, the later will appear in the
        argument declaration. default value: False

    Examples

    >>> from pyccel.ast.core import FunctionHeader
    >>> FunctionHeader('f', ['double'])
    FunctionHeader(f, [(NativeDouble(), [])])
    """

    # TODO dtypes should be a dictionary (useful in syntax)
    def __new__(cls, func, dtypes,
                results=None,
                kind='function',
                is_static=False):
        func = str(func)
        if not(iterable(dtypes)):
            raise TypeError("Expecting dtypes to be iterable.")

        if results:
            if not(iterable(results)):
                raise TypeError("Expecting results to be iterable.")

        if not isinstance(kind, str):
            raise TypeError("Expecting a string for kind.")

        if not (kind in ['function', 'procedure']):
            raise ValueError("kind must be one among {'function', 'procedure'}")

        if not isinstance(is_static, bool):
            raise TypeError('is_static must be a boolean')

        return Basic.__new__(cls, func, dtypes, results, kind, is_static)

    @property
    def func(self):
        return self._args[0]

    @property
    def dtypes(self):
        return self._args[1]

    @property
    def results(self):
        return self._args[2]

    @property
    def kind(self):
        return self._args[3]

    @property
    def is_static(self):
        return self._args[4]

    def create_definition(self):
        """Returns a FunctionDef with empy body."""
        # TODO factorize what can be factorized
        from itertools import product

        name = str(self.func)

        body      = []
        cls_name  = None
        hide      = False
        kind      = self.kind
        is_static = self.is_static
        imports   = []
        funcs = []
        dtypes = []
        for i in self.dtypes:
            if isinstance(i, UnionType):
                dtypes += [i.args]
            elif isinstance(i, dict):
                dtypes += [[i]]
            else:
                raise TypeError('element must be of type UnionType or dict')
        for args_ in product(*dtypes):
            args = []
            for i, d in enumerate(args_):
                dtype    = d['datatype']
                allocatable = d['allocatable']
                is_pointer = d['is_pointer']
                rank = d['rank']
                if rank>0 and allocatable:#case of ndarray
                    if dtype in ['int', 'double', 'float', 'complex']:
                        allocatable = True
                        dtype = 'ndarray'+dtype

                shape  = None
                if isinstance(dtype, str):
                    try:
                        dtype = datatype(dtype)
                    except:
                        #TODO check if it's a class type before
                        if isinstance(dtype, str):
                            dtype =  DataTypeFactory(str(dtype), ("_name"))()
                            is_pointer = True
                arg_name = 'arg_{0}'.format(str(i))
                arg = Variable(dtype, arg_name,
                               allocatable=allocatable, is_pointer=is_pointer,
                               rank=rank, shape=shape)
                args.append(arg)

            # ... factorize the following 2 blocks
            results = []
            for i,d_var in enumerate(self.results):
                dtype = d_var.pop('datatype')
                var = Variable(dtype, 'res_{}'.format(i), **d_var)
                results.append(var)
                # we put back dtype otherwise macro will crash when it tries to
                # call create_definition
                d_var['datatype'] = dtype

            func= FunctionDef(name, args, results, body,
                             local_vars=[],
                             global_vars=[],
                             cls_name=cls_name,
                             hide=hide,
                             kind=kind,
                             is_static=is_static,
                             imports=imports)
            funcs += [func]

        return funcs

    def to_static(self):
        """returns a static function header. needed for f2py"""
        return FunctionHeader(self.func,
                              self.dtypes,
                              self.results,
                              self.kind,
                              True)

    def __getnewargs__(self):
        """used for Pickling self."""
        args = (self.func,
                self.dtypes,
                self.results,
                self.kind,
                self.is_static,)
        return args


# TODO to be improved => use FunctionHeader
class MethodHeader(FunctionHeader):
    """Represents method header in the code.

    name: iterable
        method name as a list/tuple

    dtypes: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr)

    results: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr)

    kind: str
        'function' or 'procedure'. default value: 'function'

    is_static: bool
        True if we want to pass arrays in f2py mode. every argument of type
        array will be preceeded by its shape, the later will appear in the
        argument declaration. default value: False

    Examples

    >>> from pyccel.ast.core import MethodHeader
    >>> m = MethodHeader(('point', 'rotate'), ['double'])
    >>> m
    MethodHeader((point, rotate), [(NativeDouble(), [])], [])
    >>> m.name
    'point.rotate'
    """

    def __new__(cls, name, dtypes, results=None, kind='function', is_static=False):
        if not isinstance(name, (list, tuple)):
            raise TypeError("Expecting a list/tuple of strings.")

        if not(iterable(dtypes)):
            raise TypeError("Expecting dtypes to be iterable.")

        for d in dtypes:
            if not isinstance(d, UnionType) and not isinstance(d, dict):
                raise TypeError("Wrong element in dtypes.")

        for d in results:
            if not isinstance(d, UnionType):
                raise TypeError("Wrong element in dtypes.")


        if not isinstance(kind, str):
            raise TypeError("Expecting a string for kind.")

        if not (kind in ['function', 'procedure']):
            raise ValueError("kind must be one among {'function', 'procedure'}")

        if not isinstance(is_static, bool):
            raise TypeError('is_static must be a boolean')

        return Basic.__new__(cls, name, dtypes, results, kind, is_static)

    @property
    def name(self):
        _name = self._args[0]
        if isinstance(_name, str):
            return _name
        else:
            return '.'.join(str(n) for n in _name)

    @property
    def dtypes(self):
        return self._args[1]

    @property
    def results(self):
        return self._args[2]

    @property
    def kind(self):
        return self._args[3]

    @property
    def is_static(self):
        return self._args[4]

class ClassHeader(Header):
    """Represents class header in the code.

    name: str
        class name

    options: str, list, tuple
        a list of options

    Examples

    >>> from pyccel.ast.core import ClassHeader
    >>> ClassHeader('Matrix', ('abstract', 'public'))
    ClassHeader(Matrix, (abstract, public))
    """

    def __new__(cls, name, options):
        if not(iterable(options)):
            raise TypeError("Expecting options to be iterable.")

        return Basic.__new__(cls, name, options)

    @property
    def name(self):
        return self._args[0]

    @property
    def options(self):
        return self._args[1]


# TODO must extend Header rather than Basic
class InterfaceHeader(Basic):

    def __new__(cls, name, funcs):
        if not isinstance(name,str):
            raise TypeError('name should of type str')
        if not all([isinstance(i, str) for i in funcs]):
            raise TypeError('functions name must be of type str')
        return Basic.__new__(cls, name, funcs)


    @property
    def name(self):
        return self._args[0]

    @property
    def funcs(self):
        return self._args[1]

class MacroFunction(Header):
    """."""

    def __new__(cls, name, args, master, master_args, results=None):
        if not isinstance(name, (str, Symbol)):
            raise TypeError('name must be of type str')

        # master can be a string or FunctionDef
        if not isinstance(master, (str, FunctionDef)):
            raise ValueError('Expecting a master name of FunctionDef')

        # we sympify everything since a macro is operating on symbols
        args = [sympify(a) for a in args]
        master_args = [sympify(a) for a in master_args]

        if not(results is None):
            results = [sympify(a) for a in results]

        return Basic.__new__(cls, name, args, master, master_args, results)

    @property
    def name(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def master(self):
        return self._args[2]

    @property
    def master_arguments(self):
        return self._args[3]

    @property
    def results(self):
        return self._args[4]

    # TODO: must be moved to annotation, once we add AliasVariables
    #       this is needed if we have to create a pointer or allocate a new
    #       variable to store the result
    def apply(self, args, results=None):
        """returns the appropriate arguments."""
        # TODO improve
        if len(args) == 0:
            raise NotImplementedError('TODO')

        # ... TODO - must be a dict in order to use keywords argument (with '=')
        #            in the macro definition
        d_arguments = {}
        for (a_macro, arg) in zip(self.arguments, args):
            # TODO improve name for other Nodes
            d_arguments[a_macro.name] = arg
        argument_keys = list(d_arguments.keys())
        # ...

        # ... TODO - must be a dict in order to use keywords argument (with '=')
        #            in the macro definition
        d_results = {}
        if not(results is None) and not(self.results is None):
            for (r_macro, r) in zip(self.results, results):
                # TODO improve name for other Nodes
                d_results[r_macro.name] = r
        result_keys = list(d_results.keys())
        # ...

        # ... initialize new args with None
        newargs = []
        for i in range(0, len(self.master_arguments)):
            newargs.append(None)
        # ...

        for i,a in enumerate(self.master_arguments):
            if isinstance(a, Macro):
                new = construct_macro(a.name,
                                      d_arguments[a.argument.name])
                # TODO improve
                #      otherwise, we get the following error
                # TypeError: __new__() got multiple values for argument 'index'
                if isinstance(new, MacroShape):
                    new._index = a.index

            elif isinstance(a, MacroSymbol):
                if a.name in argument_keys:
                    new = d_arguments[a.name]

                elif a.name in result_keys:
                    new = d_results[a.name]

                elif not(a.default is None):
                    new = a.default

                else:
                    raise NotImplementedError('TODO')

            else:
                # TODO improve
                new = a

            newargs[i] = new

        return newargs
