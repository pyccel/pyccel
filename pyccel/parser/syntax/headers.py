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
from pyccel.ast.headers   import MetaVariable
from pyccel.ast.core      import FunctionDefArgument, EmptyNode
from pyccel.ast.variable  import DottedName
from pyccel.ast.literals  import LiteralString, LiteralInteger, LiteralFloat
from pyccel.ast.literals  import LiteralEllipsis, Nil
from pyccel.ast.internals import PyccelSymbol, Slice
from pyccel.ast.variable  import AnnotatedPyccelSymbol, IndexedElement
from pyccel.ast.type_annotations import SyntacticTypeAnnotation, FunctionTypeAnnotation, UnionTypeAnnotation
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

class Str(BasicStmt):
    def __init__(self, contents, **kwargs):
        self.contents = contents
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
        if isinstance(self.dtype, Str):
            dtype = LiteralString(self.dtype.contents)
        else:
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

class UnionTypeStmt(BasicStmt):
    """
    Class describing a union of possible types.

    A class object describing a union of possible types described in a type descriptor.
    These types are either VariableTypes or FuncTypes.

    Parameters
    ----------
    dtypes : list of VariableHeader | FuncHeader
        A list of the possible types described.
    **kwargs : dict
        TextX keyword arguments.
    """
    def __init__(self, dtypes, **kwargs):
        self.dtypes = list(dtypes)
        super().__init__(**kwargs)

    @property
    def expr(self):
        """
        Get the Pyccel equivalent of this object.

        Get the Pyccel equivalent of this object.
        To be removed when header support is deprecated.
        """
        dtypes = [i.expr for i in self.dtypes]
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


#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
type_classes = [UnionTypeStmt, Type, TrailerSubscriptList, FuncType, Str]
hdr_classes = [Header,
               MetavarHeaderStmt]

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

