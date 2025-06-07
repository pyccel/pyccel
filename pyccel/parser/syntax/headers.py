# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
"""
import warnings
from os.path import join, dirname

from textx import metamodel_from_file, register_language, metamodel_from_str

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.headers   import FunctionHeader, MethodHeader, Template
from pyccel.ast.headers   import MetaVariable, InterfaceHeader
from pyccel.ast.headers   import construct_macro, MacroFunction, MacroVariable
from pyccel.ast.core      import FunctionDefArgument, EmptyNode
from pyccel.ast.variable  import DottedName
from pyccel.ast.literals  import LiteralString, LiteralInteger, LiteralFloat
from pyccel.ast.literals  import LiteralTrue, LiteralFalse, LiteralEllipsis, Nil
from pyccel.ast.internals import PyccelSymbol, Slice
from pyccel.ast.variable  import AnnotatedPyccelSymbol, IndexedElement
from pyccel.ast.type_annotations import SyntacticTypeAnnotation, FunctionTypeAnnotation, UnionTypeAnnotation
from pyccel.ast.typingext import TypingFinal
from pyccel.errors.errors import Errors
from pyccel.utilities.stage import PyccelStage

DEBUG = False
errors = Errors()
pyccel_stage = PyccelStage()

class Header(object):
    """
    Class describing a Header in the grammar.

    Class describing a Header in the grammar. To be deprecated. See #1487.

    Parameters
    ----------
    statements : iterable
        A list of header statements.
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, statements = (), **kwargs):
        self.statements = statements
        super().__init__(**kwargs)

class TrailerSubscriptList(BasicStmt):
    """
    Class representing subscripts that appear trailing a type in the grammar.

    Class representing subscripts that appear trailing a type in the grammar
    in the form: `[arg1,arg2,arg3]` or `[arg1,arg2,arg3](order=c)`

    Parameters
    ----------
    args : iterable
        The arguments that appear in the subscript.
    order : str
        The order specified in brackets.
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, args, order, **kwargs):
        self.args = args
        self.order = order.capitalize() or None
        super().__init__(**kwargs)

class Type(BasicStmt):
    """
    Class representing a type in the grammar.

    Class representing a type in the grammar.

    Parameters
    ----------
    dtype : str
        The description of the base datatype.
    trailer : TrailerSubscriptList, optional
        The subscript that may appear trailing a type definition to augment its
        description.
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, dtype, trailer=None, **kwargs):
        self.dtype = dtype
        self.trailer = trailer
        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel object equivalent to this grammar object.

        Get the Pyccel object equivalent to this grammar object.
        """
        dtype = PyccelSymbol(self.dtype)
        order = None
        if self.trailer:
            args = [self.handle_trailer_arg(a) for a in self.trailer.args]
            dtype = IndexedElement(dtype, *args)
            order = self.trailer.order
        return SyntacticTypeAnnotation(dtype, order)

    def handle_trailer_arg(self, s):
        """
        Get the Pyccel object equivalent to the argument in the trailing object.

        Get the Pyccel object equivalent to the argument in the trailing object.

        Parameters
        ----------
        s : str
            The argument in the trailing section.

        Returns
        -------
        PyccelAstNode
            The Pyccel object being described.
        """
        if isinstance(s, Type):
            return s.expr
        elif s == ':':
            return Slice(None, None)
        elif s == '...':
            return LiteralEllipsis()
        else:
            raise NotImplementedError(f"Unrecognised type trailer argument : {s}")

class FuncType(BasicStmt):
    """
    Class representing a FunctionType in the grammar.

    Class representing a FunctionType in the grammar. A FunctionType is the type
    of an argument which describes a function.

    Parameters
    ----------
    args : iterable of UnionTypeStmt, optional
        A list of UnionTypeStmts describing the types of the function arguments.
    results : TypeHeader, optional
        A TypeHeader describing the type of the function result.
        (This is not a UnionTypeStmt as there cannot be multiple types for a
        given result).
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, args, results, **kwargs):
        self.args = args
        self.results = results
        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel object equivalent to this grammar object.

        Get the Pyccel object equivalent to this grammar object.
        """
        args = [a.expr for a in self.args]
        results = self.results.expr if self.results else Nil()

        return FunctionTypeAnnotation(args, results)

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
    def __init__(self, dtypes, const = False, **kwargs):
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
            dtypes = [TypingFinal(d) for d in dtypes]
        if len(dtypes)==1:
            return dtypes[0]

        return UnionTypeAnnotation(*dtypes)


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
type_classes = [UnionTypeStmt, Type, TrailerSubscriptList, FuncType]
hdr_classes = [Header,
               ShapedID,
               MetavarHeaderStmt,
               MacroStmt,
               MacroArg,
               MacroList,
               FunctionMacroStmt, StringStmt]

this_folder = dirname(__file__)

# Get meta-model from language description
types_grammar = join(this_folder, '../grammar/types.tx')
header_grammar = join(this_folder, '../grammar/headers.tx')

types_meta = metamodel_from_file(types_grammar, classes=type_classes)
register_language("types", metamodel=types_meta)

with open(header_grammar, 'r', encoding="utf-8") as f:
    grammar = f.read()
with open(types_grammar, 'r', encoding="utf-8") as f:
    grammar += f.read()

meta = metamodel_from_str(grammar, classes=hdr_classes+type_classes)
register_language("headers", metamodel=meta)

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
