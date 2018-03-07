# coding: utf-8

import os

from pyccel.parser.syntax.core import (Pyccel, ImportFromStmt, ImportAsNames, \
                                       ArithmeticExpression, Term, Atom, \
                                       ExpressionTuple, ExpressionList, \
                                       ExpressionLambda, \
                                       FactorSigned, AtomExpr, AtomExpr, Power, \
                                       FunctionHeaderStmt, ClassHeaderStmt, MethodHeaderStmt, \
                                       VariableHeaderStmt, \
                                       # statements
                                       ConstructorStmt, \
                                       DelStmt, \
                                       PassStmt, \
                                       ExpressionDict, ArgValued, \
                                       AssignStmt, AugAssignStmt, \
                                       FlowStmt, BreakStmt, ContinueStmt, \
                                       RaiseStmt, YieldStmt, ReturnStmt, \
                                       RangeStmt, \
                                       AssertStmt, IfStmt, ForStmt, \
                                       WhileStmt, WithStmt, \
                                       FunctionDefStmt, ClassDefStmt, \
                                       CallStmt, \
                                       CommentStmt, DocstringsCommentStmt, SuiteStmt, \
                                       # test bool
                                       Test, OrTest, AndTest, NotTest, Comparison, \
                                       # Trailers
                                       ArgList, \
                                       Trailer, TrailerArgList, TrailerSubscriptList, \
                                       TrailerDots, \
                                       TrailerSlice, TrailerSliceRight, \
                                       TrailerSliceLeft, TrailerSliceEmpty)

from pyccel.parser.syntax.openmp  import omp_classes
from pyccel.parser.syntax.openacc import acc_classes


from textx.metamodel import metamodel_from_file

__all__ = ["Parser", "PyccelParser", "ast_to_dict", "get_by_name"]

# ...
def get_by_name(ast, name):
    """
    Returns an object from the AST by giving its name.

    ast: textX
        the ast as given by textX
    name: str
        identifier/statement name
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
    converts the ast to a dictionary

    ast: textX
        the ast as given by textX
    """
    tokens = {}
    for token in ast.declarations:
        tokens[token.name] = token
    return tokens
# ...

class Parser(object):
    """ Class for a Parser using TextX."""
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
        return self.model.model_from_str(instructions)

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

    This is an extension of the Parser class.

    Example

    >>> code = '''
    ... n = 10
    ... for i in range(0,n):
    ...     for j in range(0,n):
    ...         x = pow(i,2) + pow(i,3) + 3*i
    ...         y = x / 3 + 2* x
    ... '''

    we first use the function *preprocess_as_str* to find indentation and add a
    TAG (indent/dedent) whenever needed.

    >>> from pyccel.codegen import preprocess_as_str
    >>> from pyccel.parser import PyccelParser
    >>> code = preprocess_as_str(code)
    >>> print code
    n = 10
    for i in range(0,n):
    indent
        for j in range(0,n):
    indent
            x = pow(i,2) + pow(i,3) + 3*i
            y = x / 3 + 2* x
    dedent
    dedent

    >>> pyccel = PyccelParser()
    >>> ast = pyccel.parse(code)

    A typical loop on the AST is the following one, note that we expect two
    statements: an assignement and a loop. The later has a body of statements,
    that contains a loop statement, and so on...

    >>> for stmt in ast.statements:
    ...     print type(stmt)
    <class 'pyccel.syntax.AssignStmt'>
    <class 'pyccel.syntax.ForStmt'>
    """
    def __init__(self, **kwargs):
        """Pyccel parser constructor.

        It takes the same arguments as the Parser class.
        """
        classes = [Pyccel, \
                   ArithmeticExpression, Term, Atom, \
                   ExpressionTuple, ExpressionList,  \
                   ExpressionLambda, \
                   FactorSigned, AtomExpr, AtomExpr, Power, \
                   FunctionHeaderStmt, ClassHeaderStmt, MethodHeaderStmt, \
                   VariableHeaderStmt, \
                   # statements
                   ConstructorStmt, \
                   AssignStmt, AugAssignStmt, \
                   DelStmt, \
                   PassStmt, \
                   ExpressionDict, ArgValued, \
                   FlowStmt, BreakStmt, ContinueStmt, \
                   RaiseStmt, YieldStmt, ReturnStmt, \
                   RangeStmt, \
                   AssertStmt, IfStmt, ForStmt, \
                   WhileStmt, WithStmt, \
                   FunctionDefStmt, ClassDefStmt, \
                   CallStmt, \
                   ImportFromStmt, ImportAsNames, \
                   CommentStmt, DocstringsCommentStmt, SuiteStmt, \
                   # test bool
                   Test, OrTest, AndTest, NotTest, Comparison, \
                   # Trailers
                   ArgList, \
                   Trailer, TrailerArgList, TrailerSubscriptList, \
                   TrailerDots, \
                   TrailerSlice, TrailerSliceRight, \
                   TrailerSliceLeft, TrailerSliceEmpty
                   ]

        classes += omp_classes
        classes += acc_classes

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(dir_path, "grammar/pyccel.tx")

        super(PyccelParser, self).__init__(filename, classes=classes)
