# coding: utf-8

# TODO must use Header.__new__ rather than Basic.__new__

from sympy.utilities.iterables import iterable
from sympy.core import Symbol
from sympy import sympify, Tuple

from .core import Basic
from .core import Variable
from .core import ValuedArgument, ValuedVariable
from .core import FunctionDef, Interface, FunctionAddress
from .core import DottedName, DottedVariable
from .datatypes import datatype, DataTypeFactory, UnionType
from .macros import Macro, MacroShape, construct_macro
from .core import local_sympify

__all__ = (
    'ClassHeader',
    'FunctionHeader',
    'Header',
    'InterfaceHeader',
    'MacroFunction',
    'MacroVariable',
    'MetaVariable',
    'MethodHeader',
    'VariableHeader',
)

#==============================================================================
class Header(Basic):
    pass

#==============================================================================
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

#==============================================================================
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

#==============================================================================
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
        True if we want to pass arrays in bind(c) mode. every argument of type
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

        if kind not in ['function', 'procedure']:
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

        def build_argument(var_name, dc):
            #Constructs an argument variable from a dictionary.
            dtype    = dc['datatype']
            allocatable = dc['allocatable']
            is_pointer = dc['is_pointer']
            precision = dc['precision']
            rank = dc['rank']
            is_const = dc['is_const']

            order = None
            shape = None
            if rank >1:
                order = dc['order']

            if isinstance(dtype, str):
                try:
                    dtype = datatype(dtype)
                except ValueError:
                    dtype = DataTypeFactory(str(dtype), ("_name"))()
            var = Variable(dtype, var_name,
                        allocatable=allocatable, is_pointer=is_pointer, is_const=is_const,
                        rank=rank, shape=shape ,order = order, precision = precision,
                        is_argument=True)
            return var


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
                # TODO  handle function as argument, which itself has a function argument
                if (d['is_func']):
                    decs = []
                    results = []
                    for dc in d['decs']:
                        var = build_argument('', dc)
                        decs.append(var)
                    for dc in d['results']:
                        var = build_argument('', dc)
                        results.append(var)
                    arg_name = 'arg_{0}'.format(str(i))
                    arg = FunctionAddress(arg_name, decs, results, [])

                else:
                    arg_name = 'arg_{0}'.format(str(i))
                    arg = build_argument(arg_name, d)
                args.append(arg)

            # ... factorize the following 2 blocks
            results = []
            for i,d_var in enumerate(self.results):
                d_var.pop('is_func')
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
                             imports=imports,
                             is_header=True)
            funcs += [func]

        return funcs

    def to_static(self):
        """returns a static function header. needed for bind(c)"""
        return FunctionHeader(self.func,
                              self.dtypes,
                              self.results,
                              self.kind,
                              True)

    def vectorize(self,index):
        """ add a dimension to one of the arguments specified by it's position"""
        types = self.dtypes
        types[index]['rank'] += 1
        types[index]['allocatable'] = True
        return FunctionHeader(self.func,
                              types,
                              self.results,
                              self.kind,
                              self.is_static)

    def __getnewargs__(self):
        """used for Pickling self."""
        args = (self.func,
                self.dtypes,
                self.results,
                self.kind,
                self.is_static,)
        return args

#==============================================================================
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
        True if we want to pass arrays in bind(c) mode. every argument of type
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

        if kind not in ['function', 'procedure']:
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

#==============================================================================
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

#==============================================================================
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

#==============================================================================
class MacroFunction(Header):
    """."""

    def __new__(cls, name, args, master, master_args, results=None):
        if not isinstance(name, (str, Symbol)):
            raise TypeError('name must be of type str or Symbol')

        # master can be a string or FunctionDef
        if not isinstance(master, (str, FunctionDef, Interface)):
            raise ValueError('Expecting a master name of FunctionDef')

        if not(results is None):
            results = [sympify(a, locals=local_sympify) for a in results]

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
        d_arguments = {}

        if len(args) > 0:

            sorted_args   = []
            unsorted_args = []
            j = -1
            for ind, i in enumerate(args):
                if not isinstance(i, ValuedArgument):
                    sorted_args.append(i)
                else:
                    j=ind
                    break
            if j>0:
                unsorted_args = args[j:]
                for i in unsorted_args:
                    if not isinstance(i, ValuedVariable):
                        raise ValueError('variable not allowed after an optional argument')

            for i in self.arguments[len(sorted_args):]:
                if not isinstance(i, ValuedVariable):
                    raise ValueError('variable not allowed after an optional argument')

            for arg,val in zip(self.arguments[:len(sorted_args)],sorted_args):
                if not isinstance(arg, Tuple):
                    d_arguments[arg.name] = val
                else:
                    if not isinstance(val, (list, Tuple,tuple)):
                        val = [val]
                    #TODO improve add more checks and generalize
                    if len(val)>len(arg):
                        raise ValueError('length mismatch of argument and its value ')
                    elif len(val)<len(arg):
                        for val_ in arg[len(val):]:
                            if isinstance(val_, ValuedVariable):
                                val +=Tuple(val_.value,)
                            else:
                                val +=Tuple(val_)

                    for arg_,val_ in zip(arg,val):
                        d_arguments[arg_.name] = val_

            d_unsorted_args = {}
            for arg in self.arguments[len(sorted_args):]:
                d_unsorted_args[arg.name] = arg.value

            for arg in unsorted_args:
                if arg.name in d_unsorted_args.keys():
                    d_unsorted_args[arg.name] = arg.value
                else:
                    raise ValueError('Unknown valued argument')
            d_arguments.update(d_unsorted_args)
            for i, arg in d_arguments.items():
                if isinstance(arg, Macro):
                    d_arguments[i] = construct_macro(arg.name,
                                      d_arguments[arg.argument.name])
                    if isinstance(arg, MacroShape):
                        d_arguments[i]._index = arg.index


        d_results = {}
        if not(results is None) and not(self.results is None):
            for (r_macro, r) in zip(self.results, results):
                # TODO improve name for other Nodes
                d_results[r_macro.name] = r
        # ...

        # ... initialize new args with None
        newargs = [None]*len(self.master_arguments)
        # ...
        argument_keys = d_arguments.keys()
        result_keys = d_results.keys()
        for i,arg in enumerate(self.master_arguments):

            if isinstance(arg, Symbol):
                if arg.name in argument_keys:
                    new = d_arguments[arg.name]
                    if isinstance(new, Symbol) and new.name in result_keys:
                        new = d_results[new.name]

                elif arg.name in result_keys:
                    new = d_results[arg.name]
                else:
                    new = arg
               #TODO uncomment later
               #     raise ValueError('Unknown variable name')
            elif isinstance(arg, Macro):
                if arg.argument.name in argument_keys:
                    new = d_arguments[arg.argument.name]
                    if isinstance(new, Symbol) and new.name in result_keys:
                        new = d_results[new.name]
                elif arg.argument.name in result_keys:
                    new = d_results[arg.argument.name]
                else:
                    raise ValueError('Unkonwn variable name')

                new = construct_macro(arg.name, new)
                if isinstance(arg, MacroShape):
                    new._index = arg.index

            newargs[i] = new
        return newargs

#==============================================================================
class MacroVariable(Header):
    """."""

    def __new__(cls, name,  master):
        if not isinstance(name, (str, Symbol, DottedName)):
            raise TypeError('name must be of type str or DottedName')


        if not isinstance(master, (str, Variable, DottedVariable)):
            raise ValueError('Expecting a master name of Variable')


        return Basic.__new__(cls, name, master)


    @property
    def name(self):
        return self._args[0]

    @property
    def master(self):
        return self._args[1]
