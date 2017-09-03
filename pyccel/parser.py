# coding: utf-8

import os

from pyccel.syntax import (Pyccel, \
                           Expression, Term, Operand, \
                           FactorSigned, FactorUnary, FactorBinary, \
                           # statements
                           DeclarationStmt, ConstructorStmt, \
                           DelStmt, \
                           PassStmt, \
                           AssignStmt, MultiAssignStmt, \
                           FlowStmt, BreakStmt, ContinueStmt, \
                           RaiseStmt, YieldStmt, ReturnStmt, \
                           IfStmt, ForStmt, FunctionDefStmt,WhileStmt,\
                           ImportFromStmt, \
                           CommentStmt, SuiteStmt, \
                           # Multi-threading
                           ThreadStmt, \
                           StencilStmt, \
                           # python standard library statements
                           PythonPrintStmt, \
                           # numpy statments
                           NumpyZerosStmt, NumpyZerosLikeStmt, \
                           NumpyOnesStmt, NumpyLinspaceStmt,NumpyArrayStmt,\
                           # test bool
                           Test, OrTest, AndTest, NotTest, Comparison, \
                           # Trailers
                           ArgList, \
                           Trailer, TrailerArgList, TrailerSubscriptList, \
                           TrailerSlice, TrailerSliceRight, TrailerSliceLeft
                           )

from pyccel.openmp.syntax import (OpenmpStmt, \
                                  ParallelStmt, \
                                  LoopStmt, \
                                  SingleStmt, \
                                  ParallelNumThreadClause, \
                                  ParallelDefaultClause, \
                                  ParallelProcBindClause, \
                                  PrivateClause, \
                                  SharedClause, \
                                  FirstPrivateClause, \
                                  LastPrivateClause, \
                                  CopyinClause, \
                                  ReductionClause, \
                                  CollapseClause, \
                                  LinearClause, \
                                  ScheduleClause, \
                                  OrderedClause, \
                                  EndConstructClause
                                 )


from textx.metamodel import metamodel_from_file

__all__ = ["PyccelParser", "ast_to_dict", "get_by_name"]

# ...
def get_by_name(ast, name):
    """
    Returns an object from the AST by giving its name.
    """
    # TODO declarations is empty for the moment
    for token in ast.declarations:
        if token.name == name:
            return token
    for token in ast.statements:
        if token.name == name:
            return token
    return None
# ...

# ...
def ast_to_dict(ast):
    """
    Returns an object from the AST by giving its name.
    """
    tokens = {}
    for token in ast.declarations:
        tokens[token.name] = token
    return tokens
# ...

class Parser(object):
    """ Class for a Parser using TextX.

    A parser can be created from a grammar (str) or a filename. It is preferable
    to specify the list classes to have more control over the abstract grammar;
    for example, to use a namespace, and to do some specific anotation.

    >>> parser = Parser(filename="gammar.tx")

    Once the parser is created, you can parse a given set of instructions by
    calling

    >>> parser.parse(["Field(V) :: u"])

    or by providing a file to parse

    >>> parser.parse_from_file("tests/inputs/1d/poisson.vl")
    """
    def __init__(self, filename, classes=None, debug=False):
        """Parser constructor.

        filename: str
            name of the file containing the abstract grammar.

        classes : list
            a list of Python classes to be used to describe the grammar. Take a
            look at TextX documentation for more details.

        debug: bool
            True if in debug mode.
        """

        # ... read the grammar from a file
        self.model = metamodel_from_file(filename, debug=debug, classes=classes)
        # ...

    def parse(self, instructions):
        """Parse a set of instructions with respect to the grammar.

        instructions: list
            list of instructions to parse.
        """
        # ... parse the DSL code
        return self.model.model_from_str(instructions)
        # ...

    def parse_from_file(self, filename):
        """Parse a set of instructions with respect to the grammar.

        filename: str
            a file containing the instructions to parse.
        """
        # ... read a DSL code
        f = open(filename)
        instructions = f.read()
        instructions.replace("\n", "")
        f.close()
        # ...

        # ... parse the DSL code
        return self.parse(instructions)
        # ...

# User friendly parser

class PyccelParser(Parser):
    """A Class for Pyccel parser.

    This is an extension of the Parser class. Additional treatment is done for
    Linear and Bilinear Forms to define their dependencies: user_fields,
    user_functions and user_constants.

    """
    def __init__(self, **kwargs):
        """Pyccel parser constructor.

        It takes the same arguments as the Parser class.
        """
        classes = [Pyccel, \
                   Expression, Term, Operand, \
                   FactorSigned, FactorUnary, FactorBinary, \
                   # statements
                   DeclarationStmt, ConstructorStmt, \
                   AssignStmt, MultiAssignStmt, \
                   DelStmt, \
                   PassStmt, \
                   FlowStmt, BreakStmt, ContinueStmt, \
                   RaiseStmt, YieldStmt, ReturnStmt, \
                   IfStmt, ForStmt, FunctionDefStmt,WhileStmt, \
                   ImportFromStmt, \
                   CommentStmt, SuiteStmt, \
                   # Multi-threading
                   ThreadStmt, \
                   StencilStmt, \
                   # python standard library statements
                   PythonPrintStmt, \
                   # numpy statments
                   NumpyZerosStmt, NumpyZerosLikeStmt, \
                   NumpyOnesStmt, NumpyLinspaceStmt,NumpyArrayStmt, \
                   # test bool
                   Test, OrTest, AndTest, NotTest, Comparison, \
                   # Trailers
                   ArgList, \
                   Trailer, TrailerArgList, TrailerSubscriptList, \
                   TrailerSlice, TrailerSliceRight, TrailerSliceLeft
                   ]

        classes += [OpenmpStmt, \
                    ParallelStmt, \
                    LoopStmt, \
                    SingleStmt, \
                    ParallelNumThreadClause, \
                    ParallelDefaultClause, \
                    ParallelProcBindClause, \
                    PrivateClause, \
                    SharedClause, \
                    FirstPrivateClause, \
                    LastPrivateClause, \
                    CopyinClause, \
                    ReductionClause, \
                    CollapseClause, \
                    LinearClause, \
                    ScheduleClause, \
                    OrderedClause, \
                    EndConstructClause
                   ]

        try:
            filename = kwargs["filename"]
        except:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(dir_path, "grammar.tx")

        super(PyccelParser, self).__init__(filename, classes=classes)

    def parse_from_file(self, filename):
        """Parse a set of instructions with respect to the grammar and returns
        the AST.

        filename: str
            a file containing the instructions to parse.
        """
        ast = super(PyccelParser, self).parse_from_file(filename)

        return ast
