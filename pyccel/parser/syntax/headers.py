# coding: utf-8

"""
"""

from os.path import join, dirname

from sympy.utilities.iterables import iterable
from sympy.core import Symbol
from sympy import sympify

from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast import FunctionHeader, ClassHeader, MethodHeader, VariableHeader
from pyccel.ast import MetaVariable , UnionType, InterfaceHeader
from pyccel.ast import construct_macro, MacroFunction, MacroVariable
from pyccel.ast import MacroSymbol, ValuedArgument
from pyccel.ast import DottedName

DEBUG = False

class Header(object):
    """Class for Header syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Header.

        """
        self.statements = kwargs.pop('statements', [])

class ListType(BasicStmt):
    """Base class representing a  ListType in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list fo str
        """
        self.dtype = kwargs.pop('dtype')

        super(ListType, self).__init__(**kwargs)

    @property
    def expr(self):
        dtypes = [str(i) for i in self.dtype]
        if not (all(dtypes[0]==i for i in dtypes)):
            raise TypeError('all element of the TypeList must have the same type')
        d_var = {}
        d_var['datatype'] = str(dtypes[0])
        d_var['rank'] = len(dtypes)
        d_var['is_pointer'] = len(dtypes)>0
        d_var['allocatable'] = False
        return d_var

class Type(BasicStmt):
    """Base class representing a header type in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Type.

        dtype: str
            variable type
        """
        self.dtype = kwargs.pop('dtype')
        self.trailer = kwargs.pop('trailer', [])

        super(Type, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dtype
        trailer = self.trailer
        if trailer:
            trailer = [str(i) for i in trailer.args]
        else:
            trailer = []
        d_var={}
        d_var['datatype']=dtype
        d_var['rank'] = len(trailer)
        d_var['allocatable'] = len(trailer)>0
        d_var['is_pointer'] = False
        return d_var

class TypeHeader(BasicStmt):
    pass

# TODO must add expr property
class UnionTypeStmt(BasicStmt):
    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list fo str
        """
        self.dtypes = kwargs.pop('dtype')

        super(UnionTypeStmt, self).__init__(**kwargs)


class HeaderResults(BasicStmt):
    """Base class representing a HeaderResults in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a HeaderResults.

        decs: list of TypeHeader
        """
        self.decs = kwargs.pop('decs')

        super(HeaderResults, self).__init__(**kwargs)

    @property
    def expr(self):
        decs = [i.expr for i in self.decs]
        return decs


class VariableHeaderStmt(BasicStmt):
    """Base class representing a header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a VariableHeader statement.
        In the case of builtin datatypes, we export a Variable

        name: str
            variable name
        dec: list, tuple
            list of argument types
        """
        self.name = kwargs.pop('name')
        self.dec  = kwargs.pop('dec')

        super(VariableHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dec.expr

        return VariableHeader(self.name, dtype)

class FunctionHeaderStmt(BasicStmt):
    """Base class representing a function header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a FunctionHeader statement

        name: str
            function name
        kind: str
            one among {function, procedure, method}
        decs: list, tuple
            list of argument types
        results: list, tuple
            list of output types
        """
        self.name = kwargs.pop('name')
        self.kind = kwargs.pop('kind', None)
        self.static = kwargs.pop('static', None)
        self.decs = kwargs.pop('decs')
        self.results = kwargs.pop('results', None)

        super(FunctionHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO: do we need dtypes and results to be attributs of the class?
        dtypes = []
        for dec in self.decs:
            if isinstance(dec,UnionTypeStmt):
                l = []
                for i in dec.dtypes:
                    l += [i.expr]
                if len(l)>1:
                    dtypes += [UnionType(l)]
                else:
                    dtypes += [l[0]]

        if self.kind is None:
            kind = 'function'
        else:
            kind = str(self.kind)

        is_static = False
        if self.static == 'static':
            is_static = True

        results = []
        if self.results:
            results = self.results.expr

        if kind == 'method':
            dtype = dtypes[0]
            if isinstance(dtype, UnionType):
                cls_instance = dtype.args[0]['datatype']
            else:
                cls_instance = dtype['datatype']
            dtypes = dtypes[1:] # remove the attribut
            kind = 'procedure'
            if results:
                kind = 'function'
            return MethodHeader((cls_instance, self.name), dtypes, [],kind=kind )
        else:
            return FunctionHeader(self.name,
                                  dtypes,
                                  results=results,
                                  kind=kind,
                                  is_static=is_static)

class ClassHeaderStmt(BasicStmt):
    """Base class representing a class header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Header statement

        name: str
            class name
        options: list, tuple
            list of class options
        """
        self.name    = kwargs.pop('name')
        self.options = kwargs.pop('options')

        super(ClassHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        options = [str(i) for i in self.options]
        return ClassHeader(self.name, options)


class MetavarHeaderStmt(BasicStmt):
    """Base class representing a metavar header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a MetavarHeader statement.

        name: str
            metavar name
        value: str
            associated value
        """
        self.name = kwargs.pop('name')
        self.value = kwargs.pop('value')

        super(MetavarHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO further checking
        name = str(self.name)
        value = self.value
        return MetaVariable(name, value)


class InterfaceStmt(BasicStmt):
      """ class represent the header interface statement"""

      def __init__(self, **kwargs):
          """
          Constructor of Interface statement

          name: str

          args: list of function names

          """

          self.name = kwargs.pop('name')
          self.args = kwargs.pop('args')
          super(InterfaceStmt, self).__init__(**kwargs)

      @property
      def expr(self):
          return InterfaceHeader(self.name, self.args)

# ...
class MacroArg(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.arg = kwargs.pop('arg')
        self.optional = kwargs.pop('optional')
        self.value = kwargs.pop('value')

        super(MacroArg, self).__init__(**kwargs)

    @property
    def expr(self):
        if self.optional:
            optional = True
        else:
            optional = False
        if self.value:
            return ValuedArgument(self.arg, self.value)

        return MacroSymbol(self.arg, is_optional=optional)

class MacroMasterArg(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.arg = kwargs.pop('arg')
        self.default = kwargs.pop('default', None)

        super(MacroMasterArg, self).__init__(**kwargs)

    @property
    def expr(self):
        arg = self.arg
        default = self.default
        if isinstance(arg, MacroStmt):
            if not(self.default is None):
                raise ValueError('No choice is allowed together with a MacroStmt')

            arg = arg.expr
        else:
            arg = Symbol(str(arg))
            if isinstance(default, MacroStmt):
                default = default.expr
            else:
                default = sympify(default)
            
            arg = MacroSymbol(arg.name, default=default)

        return arg

class ListArgsStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(ListArgsStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        args = [a.expr for a in self.args]
        return args

class ListResultsStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(ListResultsStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        return self.args

class ListAnnotatedArgsStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(ListAnnotatedArgsStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        args = [a.expr for a in self.args]
        return args

class MacroStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.arg = kwargs.pop('arg')
        self.macro = kwargs.pop('macro')
        self.parameter = kwargs.pop('parameter', None)

        super(MacroStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        name = str(self.macro)
        arg  = str(self.arg)
        parameter = self.parameter
        return construct_macro(name, arg, parameter=parameter)

# ...


class FunctionMacroStmt(BasicStmt):
    """Base class representing an alias function statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a FunctionMacroStmt statement

        name: str
            function name
        master: str
            master function name
        """
        
        self.name = tuple(kwargs.pop('name'))
        self.results = kwargs.pop('results')
        self.args = kwargs.pop('args')
        self.master_name = tuple(kwargs.pop('master_name'))
        self.master_args = kwargs.pop('master_args')

        super(FunctionMacroStmt, self).__init__(**kwargs)

    @property
    def expr(self):

        if len(self.name)>1:
            name = DottedName(*self.name)
        else:
            name = str(self.name[0])

        args = self.args
        if not (args is None):
            args = args.expr
        else:
            args = []

        if len(self.master_name)==1:
            master_name = str(self.master_name[0])
        else:
            raise NotImplementedError('TODO')

        master_args = self.master_args        
        if not (master_args is None):
            master_args = master_args.expr
        else:
            master_args = []

        results = self.results
        if not (results is None):
            results = results.expr
        else:
            results = []
       
        if len(args + master_args + results) == 0:
            return MacroVariable(name, master_name)
        if not isinstance(name, str):
            #we treat the other all the names except the last one  as arguments
            # so that we always have a name of type str
            args = list(name.name[:-1]) + list(args)
            name = name.name[-1]
        return MacroFunction(name, args, master_name, master_args,
                             results=results)


#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
hdr_classes = [Header, TypeHeader,
               Type, ListType, UnionTypeStmt,
               HeaderResults,
               FunctionHeaderStmt,
               ClassHeaderStmt,
               VariableHeaderStmt,
               MetavarHeaderStmt,
               InterfaceStmt,
               ListArgsStmt,
               ListResultsStmt,
               ListAnnotatedArgsStmt,
               MacroStmt,
               MacroArg,
               MacroMasterArg,
               FunctionMacroStmt]

def parse(filename=None, stmts=None, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, '../grammar/headers.tx')

    meta = metamodel_from_file(grammar, debug=debug, classes=hdr_classes)

    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')

    stmts = []
    for stmt in model.statements:
        e = stmt.stmt.expr
        stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts

######################
if __name__ == '__main__':
#    print(parse(stmts='#$ header variable x :: int'))
#    print(parse(stmts='#$ header variable x float [:, :]'))
#    print(parse(stmts='#$ header function f(float [:], int [:]) results(int)'))
#    print(parse(stmts='#$ header function f(float|int, int [:]) results(int)'))
#    print(parse(stmts='#$ header class Square(public)'))
#    print(parse(stmts='#$ header method translate(Point, [double], [int], int[:,:], double[:])'))
#    print(parse(stmts="#$ header metavar module_name='mpi'"))
#    print(parse(stmts='#$ header interface funcs=fun1|fun2|fun3'))
#    print(parse(stmts='#$ header function _f(int, int [:])'))
#    print(parse(stmts='#$ header macro _f(x) := f(x, x.shape)'))
#    print(parse(stmts='#$ header macro _g(x) := g(x, x.shape[0], x.shape[1])'))
#    print(parse(stmts='#$ header macro (a, b), _f(x) := f(x.shape, x, a, b)'))
#    print(parse(stmts='#$ header macro _dswap(x, incx) := dswap(x.shape, x, incx)'))
#    print(parse(stmts="#$ header macro _dswap(x, incx?) := dswap(x.shape, x, incx | 1)"))
#    print(parse(stmts='#$ header macro _dswap(x, y, incx?, incy?) := dswap(x.shape, x, incx|1, y, incy|1)'))
#    print(parse(stmts="#$ header macro _dswap(x, incx?) := dswap(x.shape, x, incx | x.shape)"))
#    print(parse(stmts='#$ header macro Point.translate(alpha, x, y) := translate(alpha, x, y)'))
    print(parse(stmts="#$ header macro _dswap(y=x, incx?) := dswap(x.shape, x, incx | x.shape)"))

