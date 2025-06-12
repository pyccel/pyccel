# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

from ..errors.errors    import Errors
from .basic             import PyccelAstNode, iterable
from .core              import Assign, FunctionCallArgument
from .core              import FunctionCall
from .internals         import PyccelSymbol, Slice
from .macros            import Macro, MacroShape, construct_macro
from .type_annotations  import SyntacticTypeAnnotation, UnionTypeAnnotation
from .variable          import DottedName, DottedVariable
from .variable          import Variable

__all__ = (
    'Header',
    'MacroFunction',
    'MacroVariable',
    'MetaVariable',
)

#==============================================================================
errors = Errors()

#==============================================================================
class Header(PyccelAstNode):
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
