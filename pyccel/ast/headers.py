# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.utilities.strings import create_incremented_string
from ..errors.errors    import Errors
from ..errors.messages  import TEMPLATE_IN_UNIONTYPE
from .basic             import Basic, iterable
from .core              import Assign, FunctionCallArgument
from .core              import FunctionDef, FunctionCall, FunctionAddress
from .core              import FunctionDefArgument, FunctionDefResult
from .datatypes         import datatype, DataTypeFactory, UnionType, default_precision
from .internals         import PyccelSymbol, Slice
from .macros            import Macro, MacroShape, construct_macro
from .variable          import DottedName, DottedVariable
from .variable          import Variable

__all__ = (
    'ClassHeader',
    'FunctionHeader',
    'Header',
    'InterfaceHeader',
    'MacroFunction',
    'MacroVariable',
    'MetaVariable',
    'MethodHeader',
    'Template',
    'VariableHeader',
)

#==============================================================================
errors = Errors()

#==============================================================================
class Header(Basic):
    __slots__ = ()
    _attribute_nodes = ()

#==============================================================================
class MetaVariable(Header):
    """Represents the MetaVariable."""
    __slots__ = ('_name', '_value')

    def __init__(self, name, value):
        if not isinstance(name, str):
            raise TypeError('name must be of type str')

        # TODO check value
        self._name  = name
        self._value = value

        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable that can be called
           to create the initial version of the object
           and its arguments
        """
        return (self.__class__, (self.name, self.value))

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
    __slots__ = ('_name','_dtypes')

    def __init__(self, name, dtypes):
        if not(isinstance(dtypes, dict)):
            raise TypeError("Expecting dtypes to be a dict.")

        self._name   = name
        self._dtypes = dtypes

        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def dtypes(self):
        return self._dtypes

    def __reduce_ex__(self, i):
        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable that can be called
           to create the initial version of the object
           and its arguments
        """
        return (self.__class__, (self.name, self.dtypes))

#==============================================================================
class Template(Header):
    """Represents a template.

    Parameters
    ----------
    name: str
        The name of the template.

    dtypes: iterable
        The types the template represents

    Examples
    --------
    >>> from pyccel.ast.headers import Template
    >>> d_var0 = {'datatype': 'int', 'rank': 0, 'memory_handling': 'stack',\
    >>>        'precision': 8, 'is_func': False, 'is_const': False}
    >>> d_var1 = {'datatype': 'int', 'rank': 0, 'memory_handling': 'stack',\
    >>>        'precision': 8, 'is_func': False, 'is_const': False}
    >>> T = Template('T', [d_var0, d_var1])
    """
    __slots__ = ('_name','_dtypes')

    def __init__(self, name, dtypes):
        super().__init__()
        self._name = name
        self._dtypes = dtypes

    @property
    def name(self):
        "The name of the template."
        return self._name

    @property
    def dtypes(self):
        "Types the template represents."
        return self._dtypes

    def __reduce_ex__(self, i):

        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments
        """
        return (self.__class__, (self.name, self.dtypes))

#==============================================================================
class FunctionHeader(Header):
    """Represents function/subroutine header in the code.

    Parameters
    ----------
    name: str
        function/subroutine name

    dtypes: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr, allocatable)

    results: tuple/list
        a list of datatypes. an element of this list can be str/DataType of a
        tuple (str/DataType, attr, allocatable)

    is_static: bool
        True if we want to pass arrays in bind(c) mode. every argument of type
        array will be preceeded by its shape, the later will appear in the
        argument declaration. default value: False

    Examples
    --------

    >>> from pyccel.ast.headers import FunctionHeader
    >>> FunctionHeader('f', ['double'])
    FunctionHeader(f, [(NativeDouble(), [])])
    """
    __slots__ = ('_name','_dtypes','_results','_is_static')

    # TODO dtypes should be a dictionary (useful in syntax)
    def __init__(self, name, dtypes,
                 results=None,
                 is_static=False):

        if not(iterable(dtypes)):
            raise TypeError("Expecting dtypes to be iterable.")

        if results:
            if not(iterable(results)):
                raise TypeError("Expecting results to be iterable.")

        if not isinstance(is_static, bool):
            raise TypeError('is_static must be a boolean')

        self._name      = name
        self._dtypes    = dtypes
        self._results   = results
        self._is_static = is_static
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def results(self):
        return self._results

    @property
    def is_static(self):
        return self._is_static

    def create_definition(self, templates = (), is_external=False):
        """Returns a FunctionDef with empy body."""
        # TODO factorize what can be factorized
        from itertools import product

        name = self.name

        body      = []
        cls_name  = None
        is_static = self.is_static
        used_names = set(name)
        imports   = []
        funcs = []
        dtypes = []

        def build_argument(var_name, dc):
            #Constructs an argument variable from a dictionary.
            dtype    = dc['datatype']
            memory_handling = dc['memory_handling']
            precision = dc['precision']
            rank = dc['rank']
            is_const = dc['is_const']

            order = None
            shape = None
            annotation = None

            if rank and precision == -1:
                precision = default_precision[dtype]

            if rank >1:
                order = dc['order']

            if isinstance(dtype, str):
                annotation = dtype
                try:
                    dtype = datatype(dtype)
                except ValueError:
                    dtype = DataTypeFactory(str(dtype), ("_name"))()
            var = Variable(dtype, var_name,
                           memory_handling=memory_handling, is_const=is_const,
                           rank=rank, shape=shape ,order=order, precision=precision,
                           is_argument=True, is_temp=True)

            return FunctionDefArgument(var, annotation = annotation)

        def process_template(signature, Tname, d_type):
            #Replaces templates named Tname inside signature, with the given type.
            new_sig = tuple(d_type if 'datatype' in t and t['datatype'] == Tname\
                            else t for t in signature)
            return new_sig

        def find_templates(signature, templates):
            #Creates a dictionary of only used templates in signature.
            new_templates = {d_type['datatype']:templates[d_type['datatype']]\
                             for d_type in signature\
                             if 'datatype' in d_type and d_type['datatype'] in templates}
            return new_templates

        for i in self.dtypes:
            if isinstance(i, UnionType):
                for d_type in i.args:
                    if d_type['datatype'] in templates:
                        errors.report(TEMPLATE_IN_UNIONTYPE,
                                      symbol=self.name,
                                      severity='error')
                dtypes += [i.args]
            elif isinstance(i, dict):
                dtypes += [[i]]
            else:
                raise TypeError('element must be of type UnionType or dict')

        #TODO: handle the case of functions arguments

        signatures = list(product(*dtypes))
        new_templates = find_templates(signatures[0], templates)

        for tmplt in new_templates:
            signatures = tuple(process_template(s, tmplt, d_type)\
                               for s in signatures for d_type in new_templates[tmplt].dtypes)

        for args_ in signatures:
            args = []
            for i, d in enumerate(args_):
                # TODO  handle function as argument, which itself has a function argument
                if (d['is_func']):
                    decs = []
                    results = []
                    _count = 0
                    for dc in d['decs']:
                        _name, _count = create_incremented_string(used_names, 'in', _count)
                        var = build_argument(_name, dc)
                        decs.append(var)
                    _count = 0
                    for dc in d['results']:
                        _name, _count = create_incremented_string(used_names, 'out', _count)
                        var = build_argument(_name, dc)
                        results.append(FunctionDefResult(var.var))
                    arg_name = 'arg_{0}'.format(str(i))
                    arg = FunctionDefArgument(FunctionAddress(arg_name, decs, results, []))

                else:
                    arg_name = 'arg_{0}'.format(str(i))
                    arg = build_argument(arg_name, d)
                args.append(arg)

            # ... factorize the following 2 blocks
            results = []
            for i,d_var in enumerate(self.results):
                is_func = d_var.pop('is_func')
                dtype = d_var.pop('datatype')
                var = Variable(dtype, 'res_{}'.format(i), **d_var, is_temp = True)
                results.append(FunctionDefResult(var, annotation = str(dtype)))
                # we put back dtype otherwise macro will crash when it tries to
                # call create_definition
                d_var['datatype'] = dtype
                d_var['is_func'] = is_func

            func= FunctionDef(name, args, results, body,
                              global_vars=[],
                              cls_name=cls_name,
                              is_static=is_static,
                              imports=imports,
                              is_header=True,
                              is_external=is_external)
            funcs += [func]

        return funcs

    def to_static(self):
        """returns a static function header. needed for bind(c)"""
        return FunctionHeader(self.name,
                              self.dtypes,
                              self.results,
                              True)

    def vectorize(self,index):
        """ add a dimension to one of the arguments specified by it's position"""
        types = self.dtypes
        types[index]['rank'] += 1
        types[index]['memory_handling'] = 'heap'
        return FunctionHeader(self.name,
                              types,
                              self.results,
                              self.is_static)


    def __reduce_ex__(self, i):

        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments
        """

        args = (self.name,
                self.dtypes,
                self.results,
                self.is_static,)
        return (self.__class__, args)


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

    is_static: bool
        True if we want to pass arrays in bind(c) mode. every argument of type
        array will be preceeded by its shape, the later will appear in the
        argument declaration. default value: False

    Examples

    >>> from pyccel.ast.headers import MethodHeader
    >>> m = MethodHeader(('point', 'rotate'), ['double'])
    >>> m
    MethodHeader((point, rotate), [(NativeDouble(), [])], [])
    >>> m.name
    'point.rotate'
    """
    __slots__ = ()

    def __init__(self, name, dtypes, results=None, is_static=False):
        if not isinstance(name, (list, tuple)):
            raise TypeError("Expecting a list/tuple of strings.")
        name      = '.'.join(str(n) for n in name)

        if not(iterable(dtypes)):
            raise TypeError("Expecting dtypes to be iterable.")

        for d in dtypes:
            if not isinstance(d, UnionType) and not isinstance(d, dict):
                raise TypeError("Wrong element in dtypes.")

        for d in results:
            if not isinstance(d, UnionType):
                raise TypeError("Wrong element in dtypes.")


        if not isinstance(is_static, bool):
            raise TypeError('is_static must be a boolean')

        super().__init__(name, dtypes, results, is_static)

    def __reduce_ex__(self, i):

        """ Used by pickle to create an object of this class.

          Parameters
          ----------

          i : int
           protocol

          Results
          -------

          out : tuple
           A tuple of two elements
           a callable function that can be called
           to create the initial version of the object
           and its arguments
        """

        args = (self.name.split('.'),
                self.dtypes,
                self.results,
                self.is_static,)
        return (self.__class__, args)

#==============================================================================
class ClassHeader(Header):
    """Represents class header in the code.

    name: str
        class name

    options: str, list, tuple
        a list of options

    Examples

    >>> from pyccel.ast.headers import ClassHeader
    >>> ClassHeader('Matrix', ('abstract', 'public'))
    ClassHeader(Matrix, (abstract, public))
    """
    __slots__ = ('_name','_options')

    def __init__(self, name, options):
        if not(iterable(options)):
            raise TypeError("Expecting options to be iterable.")

        self._name    = name
        self._options = options

        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def options(self):
        return self._options

#==============================================================================
class InterfaceHeader(Header):
    """Represents an interface header in the code.

    Parameters
    ----------
    name: str
        the name used to call the functions

    funcs: tuple/list of str
        a list containing the names of the functions available via this
        interface

    Examples
    --------
    >>> from pyccel.ast.headers import InterfaceHeader
    >>> m = InterfaceHeader('axpy',('daxpy', 'saxpy'))
    >>> m
    InterfaceHeader('axpy',('daxpy', 'saxpy'))
    >>> m.name
    'axpy'
    """
    __slots__ = ('_name','_funcs')

    def __init__(self, name, funcs):
        if not isinstance(name,str):
            raise TypeError('name should of type str')
        if not all([isinstance(i, str) for i in funcs]):
            raise TypeError('functions name must be of type str')
        self._name  = name
        self._funcs = funcs
        super().__init__()


    @property
    def name(self):
        return self._name

    @property
    def funcs(self):
        return self._funcs

#==============================================================================
class MacroFunction(Header):
    """Represents a macro function header in the code.
    A macro function header maps a defined function name, set of arguments
    and results to a different function call. This is notably useful when
    mapping python functions to fortran functions which do the same thing but
    do not have the same set of arguments/results.

    Parameters
    ----------

    name: str
        the name used to call the function

    args: iterable
        The python arguments of the macro

    master: str
        The name of the function in Fortran

    master_args: iterable
        The Fortran arguments of the macro

    results: iterable
        The python results of the macro

    results_shapes: iterable
        A list of shapes of the results
    """

    __slots__ = ('_name','_arguments','_master','_master_arguments',
                 '_results','_copies_required', '_results_shapes')

    def __init__(self, name, args, master, master_args, results=None, results_shapes=None):
        if not isinstance(name, str):
            raise TypeError('name must be of type str or PyccelSymbol')

        # master can be a string or FunctionCall
        if not isinstance(master, (str, FunctionCall)):
            raise ValueError('Expecting a function name, or a FunctionCall')

        self._name              = name
        self._arguments         = args
        self._master            = master
        self._master_arguments  = master_args
        self._results           = results
        self._results_shapes    = results_shapes
        self._copies_required   = [a in self._results for a in self._arguments]
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def arguments(self):
        return self._arguments

    @property
    def master(self):
        return self._master

    @property
    def master_arguments(self):
        return self._master_arguments

    @property
    def results(self):
        return self._results

    @property
    def results_shapes(self):
        """an iterable of the shapes of the results"""
        return self._results_shapes

    def link_args(self, args):
        """links macro arguments to the appropriate functioncall args

        Parameters
        ----------
        args: iterable
            The arguments provided at function call

        Results
        -------
        d_arguments: dictionary
            A Dictionary, where the keys are arguments from the macro
            and values are arguments provided at function call
        """

        d_arguments = {}

        if len(args) > 0:

            unsorted_args = []
            n_sorted = len(args)
            for ind, (arg, val) in enumerate(zip(self.arguments, args)):
                if not val.has_keyword:
                    name = str(arg) if isinstance(arg, PyccelSymbol) \
                            else arg.name
                    d_arguments[name] = val.value
                else:
                    n_sorted=ind
                    break

            unsorted_args = args[n_sorted:]
            for i in unsorted_args:
                if not i.has_keyword:
                    errors.report("Positional argument not allowed after an optional argument",
                            symbol=i,
                            severity='fatal')

            d_unsorted_args = {}
            for arg in self.arguments[n_sorted:]:
                d_unsorted_args[arg.name] = arg.value

            for arg in unsorted_args:
                if arg.keyword in d_unsorted_args.keys():
                    d_unsorted_args[arg.keyword] = arg.value
                else:
                    raise ValueError('Unknown valued argument')

            d_arguments.update(d_unsorted_args)

            for i, arg in d_arguments.items():
                if isinstance(arg, Macro):
                    d_arguments[i] = construct_macro(arg.name,
                                                     d_arguments[arg.argument])
                    if isinstance(arg, MacroShape):
                        d_arguments[i]._index = arg.index
        return d_arguments


    def get_results_shapes(self, args):
        """replace elements of the shape with appropriate values

        Parameters
        ----------
        args: iterable
            The arguments provided at function call

        Results
        -------
        results_shapes: iterable
            List of shapes after replacing variables indicated
            in the macro if they exist, with the appropriate variables
            from args.
        """

        d_arguments = self.link_args(args)
        results_shapes = []
        for result, shape in zip(self.results, self.results_shapes):
            newargs = []

            for i, arg in enumerate(shape):
                if str(arg) == ':':
                    try:
                        new = MacroShape(d_arguments[result], i)
                    except KeyError:
                        msg = "Shape needs to be provided explicitly as it cannot be deduced"
                        errors.report(msg, symbol=result,
                                      severity='error')

                elif isinstance(arg, PyccelSymbol):
                    new = d_arguments.get(arg, arg)
                elif isinstance(arg, Macro):
                    if arg.argument in d_arguments:
                        new = d_arguments[arg.argument]
                    else:
                        raise ValueError('Unknown variable name')

                    if isinstance(arg, MacroShape):
                        new = MacroShape(new, arg.index)
                    else:
                        new = construct_macro(arg.name, new)
                else:
                    new=arg

                newargs.append(new)
            newargs = None if len(newargs) == 0 else tuple(newargs)
            results_shapes.append(newargs)
        return results_shapes

    def apply(self, args, results=None):
        """Converts the arguments provided to the macro to arguments
        appropriate for the FunctionCall to the associated function
        """

        d_arguments = self.link_args(args)
        d_results = {}
        if not(results is None) and not(self.results is None):
            for (r_macro, r) in zip(self.results, results):
                # TODO improve name for other Nodes
                d_results[r_macro] = r
        # ...

        # ... initialize new args with None
        newargs = [None]*len(self.master_arguments)
        # ...
        argument_keys = d_arguments.keys()
        result_keys = d_results.keys()
        for i,arg in enumerate(self.master_arguments):

            value = arg.value
            if isinstance(value, Variable):
                arg_name = value.name
                if arg_name in result_keys:
                    new_value = d_results[arg_name]

                elif arg_name in argument_keys:
                    new_value = d_arguments[arg_name]

                else:
                    raise ValueError('Missing argument')

            elif isinstance(value, Macro):
                if value.argument in result_keys:
                    new_value = d_results[value.argument]
                elif value.argument in argument_keys:
                    new_value = d_arguments[value.argument]
                else:
                    raise ValueError('Unknown variable name')

                if isinstance(value, MacroShape):
                    new_value = MacroShape(new_value, value.index)
                else:
                    new_value = construct_macro(value.name, new_value)
            else:
                new_value = value
            newargs[i] = FunctionCallArgument(new_value)
        return newargs

    def make_necessary_copies(self, args, results):
        """ Copy any arguments provided in python which the macro
        definition indicates should match the results into the
        corresponding result.

        Parameters
        ----------
        args    : list of Variables
                   The arguments passed to the macro in the python code
        results : list of Variables
                  The results collected from the macro in the python code

        Results
        -------
        expr       : list of Assigns
                      Any Assigns necessary before the function (result of the
                      macro expansion) returned by the apply function is called
        """
        expr = []
        for a, func_a in zip(args, self.arguments):
            arg = a.value
            func_arg = func_a.var
            if func_arg in self.results and arg.rank > 0:
                r = results[self.results.index(func_arg)]
                if arg != r:
                    slices = [Slice(None,None)]*arg.rank
                    expr.append(Assign(r[slices], arg[slices]))
                    arg = r

        return expr


#==============================================================================
class MacroVariable(Header):
    """."""
    __slots__ = ('_name','_master')

    def __init__(self, name,  master):
        if not isinstance(name, (str, DottedName)):
            raise TypeError('name must be of type str or DottedName')


        if not isinstance(master, (str, Variable, DottedVariable)):
            raise ValueError('Expecting a master name of Variable')

        self._name   = name
        self._master = master

        super().__init__()


    @property
    def name(self):
        return self._name

    @property
    def master(self):
        return self._master
