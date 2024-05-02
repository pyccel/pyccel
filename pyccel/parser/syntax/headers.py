# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
"""
import warnings
from os.path import join, dirname

from textx.metamodel import metamodel_from_file

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.headers   import FunctionHeader, MethodHeader, Template
from pyccel.ast.headers   import MetaVariable , UnionType, InterfaceHeader
from pyccel.ast.headers   import construct_macro, MacroFunction, MacroVariable
from pyccel.ast.core      import FunctionDefArgument, EmptyNode
from pyccel.ast.variable  import DottedName
from pyccel.ast.datatypes import dtype_and_precision_registry as dtype_registry, default_precision
from pyccel.ast.datatypes import NativeNumeric
from pyccel.ast.literals  import LiteralString, LiteralInteger, LiteralFloat
from pyccel.ast.literals  import LiteralTrue, LiteralFalse
from pyccel.ast.internals import PyccelSymbol
from pyccel.ast.variable  import AnnotatedPyccelSymbol
from pyccel.ast.type_annotations import SyntacticTypeAnnotation
from pyccel.errors.errors import Errors
from pyccel.utilities.stage import PyccelStage

DEBUG = False
errors = Errors()
pyccel_stage = PyccelStage()

class Header(object):
    """Class for Header syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Header.

        """
        self.statements = kwargs.pop('statements', [])

class FuncType(BasicStmt):
    """Base class representing a  FunctionType in the grammar."""
    def __init__(self, **kwargs):
        self.decs = kwargs.pop('decs')
        self.results = kwargs.pop('results')

        super().__init__(**kwargs)

    @property
    def expr(self):
        decs = []
        if self.decs:
            decs = [i.expr for i in self.decs]

        results = []
        if self.results:
            results = [i.expr for i in self.results]

        d_var = {}
        d_var['decs'] = decs
        d_var['results'] = results
        d_var['is_func'] = True

        return d_var

class TemplateStmt(BasicStmt):
    """
    Base class representing a template in the grammar.

    Base class representing a template in the grammar.
    To be removed when header support is deprecated.

    Parameters
    ----------
    name : str
        The name of the template type symbol.
    dtypes : list of str
        A list of the types that the template describes.
    **kwargs : dict
        The textx arguments.
    """
    def __init__(self, *, name, dtypes, **kwargs):
        self.dtypes = dtypes
        self.name   = name
        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel equivalent of this object.

        Get the Pyccel equivalent of this object.
        """
        if any(isinstance(d_type, FuncType) for d_type in self.dtypes):
            msg = 'Functions in a template are not supported yet'
            errors.report(msg,
                        severity='error')
            return EmptyNode()

        dtypes = {SyntacticTypeAnnotation.build_from_textx(t)  for t in self.dtypes}
        return Template(self.name, dtypes)

class VariableType(BasicStmt):
    """
    Base class representing a header type in the grammar.

    Base class representing a header type in the grammar.

    Parameters
    ----------
    dtype : str
        The variable type.

    prec : int, optional
        The precision of the object.

    trailer : iterable, TrailerSubscriptsList
        An object created by textx describing the trailing decorators of the
        type. The number of elements is equal to the rank. The order is also
        described when the iterable is non-empty.

    **kwargs : dict
        The textx arguments.
    """

    def __init__(self, dtype, prec = None, trailer = (), **kwargs):
        self.dtype   = dtype
        self.prec    = prec
        if trailer:
            self.trailer = trailer
        else:
            self.trailer = []

        super().__init__(**kwargs)

class ShapedID(BasicStmt):
    """class representing a ShapedID in the grammar.

    Parameters
    ----------
    name: str
        Name of the variable result

    shape: list
        A list representing the shape of the result
    """

    def __init__(self, **kwargs):
        self._name   = kwargs.pop('name')
        self._shape  = kwargs.pop('shape', [])

        super().__init__(**kwargs)

    @property
    def expr(self):
        """Returns a dictionary containing name and shape of result"""

        shape = [i.expr if isinstance(i, MacroStmt) else PyccelSymbol(i) for i in self._shape]
        d_var = {'name': self._name, 'shape': shape}

        return d_var

class StringStmt(BasicStmt):
    """
    Class describing a string in a macro.

    Class describing a string in a macro.
    To be removed when macro support is deprecated.

    Parameters
    ----------
    arg : str
        The string.
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, arg, **kwargs):
        self.arg = arg
        super().__init__(**kwargs)
    @property
    def expr(self):
        return LiteralString(str(self.arg))

class UnionTypeStmt(BasicStmt):
    """
    Class describing a union of possible types.

    A class object describing a union of possible types described in a type descriptor.
    These types are either VariableTypes or FuncTypes.

    Parameters
    ----------
    dtypes : list of VariableHeader | FuncHeader
        A list of the possible types described.
    const : bool
        A boolean indicating if the generated object will be constant or
        modifiable.
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, dtypes, const = None, **kwargs):
        self.dtypes = list(dtypes)
        self.const = const

        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel equivalent of this object.

        Get the Pyccel equivalent of this object.
        To be removed when header support is deprecated.
        """
        dtypes = [i.expr for i in self.dtypes]
        if self.const:
            for d_type in dtypes:
                d_type["is_const"] = True
        if len(dtypes)==1:
            return dtypes[0]
        if any(isinstance(d_type, FuncType) for d_type in self.dtypes):
            msg = 'Functions in a uniontype are not supported yet'
            errors.report(msg,
                        severity='error')
            return EmptyNode()

        possible_dtypes = {tuple(t.items())  for t in dtypes}
        dtypes = [dict(d_type) for d_type in possible_dtypes]
        return UnionType(dtypes)

class HeaderResults(BasicStmt):
    """
    Base class representing a HeaderResults in the grammar.

    Base class representing a HeaderResults in the grammar.
    To be removed when header support is deprecated.

    Parameters
    ----------
    decs : list of VariableHeader | FuncHeader
        List of TypeHeaders defining the return type(s).
    **kwargs : dict
        TextX keyword arguments.
    """

    def __init__(self, decs, **kwargs):
        self.decs = decs

        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel equivalent of this object.

        Get the Pyccel equivalent of this object.
        """
        decs = SyntacticTypeAnnotation.build_from_textx(self.decs)
        return decs


class VariableHeaderStmt(BasicStmt):
    """
    Base class representing a header statement in the grammar.

    Base class representing a header statement in the grammar.
    To be removed when header support is deprecated.

    Parameters
    ----------
    name : str
        Variable name.
    dec : list, tuple
        List of argument types.
    **kwargs : dict
        TextX keyword arguments.
    """

    def __init__(self, name, dec, **kwargs):
        self.name = name
        self.dec  = dec

        super(VariableHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Convert a VariableHeaderStmt generated by TextX to a Pyccel AST object.

        Convert a VariableHeaderStmt generated by TextX to a Pyccel AST object.
        """
        warnings.warn("Support for specifying types via headers will be removed in " +
                      "a future version of Pyccel. This annotation may be unnecessary " +
                      "in your code. If you find it is necessary please open a discussion " +
                      "at https://github.com/pyccel/pyccel/discussions so we do not " +
                      "remove support until an alternative is in place.", FutureWarning)
        dtype = SyntacticTypeAnnotation.build_from_textx(self.dec)

        return AnnotatedPyccelSymbol(self.name, annotation=dtype)

class FunctionHeaderStmt(BasicStmt):
    """
    Base class representing a function header statement in the grammar.

    Base class representing a function header statement in the grammar.
    To be removed when header support is deprecated.

    Parameters
    ----------
    name : str
        Function name.
    decs : list, tuple
        List of argument types.
    kind : str
        One among {function, method}.
    static : bool
        Indicates if the function is a static function.
    results : list, tuple
        List of output types.
    **kwargs : dict
        Arguments defined by textx.
    """

    def __init__(self, name, decs, kind=None, static = None, results=None, **kwargs):
        self.name = name
        self.kind = kind
        self.static = static
        self.decs = decs
        self.results = results

        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel equivalent of this object.

        Get the Pyccel equivalent of this object.
        """
        dtypes = SyntacticTypeAnnotation.build_from_textx(self.decs)

        if self.kind is None:
            kind = 'function'
        else:
            kind = str(self.kind)

        is_static = self.static

        results = []
        if self.results:
            results = self.results.expr

        if kind == 'method':
            return MethodHeader(self.name, dtypes,
                                  results=results,
                                  is_static=is_static)
        else:
            return FunctionHeader(self.name,
                                  dtypes,
                                  results=results,
                                  is_static=is_static)


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
        self.value = kwargs.pop('value',None)

        super(MacroArg, self).__init__(**kwargs)

    @property
    def expr(self):
        arg_ = self.arg
        if isinstance(arg_, MacroList):
            return tuple(arg_.expr)
        arg = PyccelSymbol(arg_)
        value = self.value
        if not(value is None):
            if isinstance(value, (MacroStmt,StringStmt)):
                value = value.expr
            return FunctionDefArgument(arg, value=value)
        return arg


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
        arg  = PyccelSymbol(self.arg)
        parameter = self.parameter
        return construct_macro(name, arg, parameter=parameter)

# ...

class MacroList(BasicStmt):
    """ reresent a MacroList statement"""
    def __init__(self, **kwargs):
        ls = []
        for i in kwargs.pop('ls'):
            if isinstance(i, MacroArg):
                ls.append(i.expr)
            else:
                ls.append(i)
        self.ls = ls

        super(MacroList, self).__init__(**kwargs)

    @property
    def expr(self):
        return self.ls


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

        self.dotted_name = tuple(kwargs.pop('dotted_name'))
        self.results = kwargs.pop('results', [])
        self.args = kwargs.pop('args')
        self.master_name = tuple(kwargs.pop('master_name'))
        self.master_args = kwargs.pop('master_args')

        super(FunctionMacroStmt, self).__init__(**kwargs)

    @property
    def expr(self):

        if len(self.dotted_name)>1:
            name = DottedName(*self.dotted_name)
        else:
            name = str(self.dotted_name[0])

        args = []
        for i in self.args:
            if isinstance(i, MacroArg):
                args.append(i.expr)
            else:
                raise TypeError('argument must be of type MacroArg')


        if len(self.master_name)==1:
            master_name = str(self.master_name[0])
        else:
            raise NotImplementedError('TODO')

        master_args = []
        for i in self.master_args:
            if isinstance(i, MacroStmt):
                master_args.append(i.expr)
            elif isinstance(i, str):
                master_args.append(PyccelSymbol(i))
            elif isinstance(i, int):
                master_args.append(LiteralInteger(i))
            elif isinstance(i, float):
                master_args.append(LiteralFloat(i))
            elif i is True:
                master_args.append(LiteralTrue())
            elif i is False:
                master_args.append(LiteralFalse())
            else:
                NotImplementedError("Unrecognised macro argument type")

        results = [PyccelSymbol(r.expr['name']) for r in self.results]
        results_shapes = [r.expr['shape'] for r in self.results]

        if len(args + master_args + results) == 0:
            return MacroVariable(name, master_name)

        if not isinstance(name, str):
            #we treat the other all the names except the last one  as arguments
            # so that we always have a name of type str
            args = list(name.name[:-1]) + list(args)
            name = name.name[-1]
        return MacroFunction(name, args, master_name, master_args, results=results,
                             results_shapes=results_shapes)


#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
hdr_classes = [Header, VariableType, UnionTypeStmt, FuncType,
               ShapedID,
               HeaderResults,
               FunctionHeaderStmt,
               TemplateStmt,
               VariableHeaderStmt,
               MetavarHeaderStmt,
               InterfaceStmt,
               MacroStmt,
               MacroArg,
               MacroList,
               FunctionMacroStmt,StringStmt]

this_folder = dirname(__file__)

# Get meta-model from language description
grammar = join(this_folder, '../grammar/headers.tx')
types_grammar = join(this_folder, '../grammar/types.tx')

meta = metamodel_from_file(grammar, classes=hdr_classes)
types_meta = metamodel_from_file(types_grammar)

def parse(filename=None, stmts=None):
    """ Parse header pragmas

      Parameters
      ----------

      filename: str

      stmts   : list

      Results
      -------

      stmts  : list

    """
    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')
    # Ensure correct stage
    pyccel_stage.set_stage('syntactic')

    stmts = []
    for stmt in model.statements:
        e = stmt.stmt.expr
        stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
if __name__ == '__main__':
    print(parse(stmts='#$ header variable x :: int'))
    print(parse(stmts='#$ header variable x float [:, :]'))
    print(parse(stmts='#$ header function f(float [:], int [:]) results(int)'))
    print(parse(stmts='#$ header function f(float|int, int [:]) results(int)'))
    print(parse(stmts='#$ header method translate(Point, [double], [int], int[:,:], double[:])'))
    print(parse(stmts="#$ header metavar module_name='mpi'"))
    print(parse(stmts='#$ header interface funcs=fun1|fun2|fun3'))
    print(parse(stmts='#$ header function _f(int, int [:])'))
    print(parse(stmts='#$ header macro _f(x) := f(x, x.shape)'))
    print(parse(stmts='#$ header macro _g(x) := g(x, x.shape[0], x.shape[1])'))
    print(parse(stmts='#$ header macro (a, b), _f(x) := f(x.shape, x, a, b)'))
    print(parse(stmts='#$ header macro _dswap(x, incx) := dswap(x.shape, x, incx)'))
    print(parse(stmts="#$ header macro _dswap(x, incx=1) := dswap(x.shape, x, incx)"))
    print(parse(stmts='#$ header macro _dswap(x, y, incx=1, incy=1) := dswap(x.shape, x, incx, y, incy)'))
    print(parse(stmts="#$ header macro _dswap(x, incx=x.shape) := dswap(x.shape, x, incx)"))
    print(parse(stmts='#$ header macro Point.translate(alpha, x, y) := translate(alpha, x, y)'))
    print(parse(stmts="#$ header macro _dswap([data,dtype=data.dtype,count=count.dtype], incx=y.shape,M='M',d=incx) := dswap(y.shape, y, incx)"))
    print(parse(stmts='#$ header function _f(int, int [:,:](order = F))'))
    print(parse(stmts='#$ header function _f(int, int [:,:])'))
