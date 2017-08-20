# coding: utf-8

import os
from sympy import Symbol, sympify

from vale.utilities import grad, d_var, inner, outer, cross, dot
from pyccel.syntax import (Pyccel, \
                           Expression, Term, Operand, \
                           FactorSigned, FactorUnary, FactorBinary, \
                           # statements
                           DeclarationStmt, \
                           DelStmt, \
                           PassStmt, \
                           AssignStmt, \
                           FlowStmt, BreakStmt, ContinueStmt, \
                           RaiseStmt, YieldStmt, ReturnStmt, \
                           IfStmt, ForStmt, FunctionDefStmt, \
                           ImportFromStmt, \
                           # numpy statments
                           NumpyZerosStmt, NumpyZerosLikeStmt, \
                           NumpyOnesStmt, NumpyLinspaceStmt \
                           )

from textx.metamodel import metamodel_from_str

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
    def __init__(self, grammar=None, filename=None, \
                 classes=None):
        """Parser constructor.

        grammar : str
            abstract grammar describing the DSL.

        filename: str
            name of the file containing the abstract grammar.

        classes : list
            a list of Python classes to be used to describe the grammar. Take a
            look at TextX documentation for more details.
        """

        _grammar = grammar

        # ... read the grammar from a file
        if not (filename is None):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(dir_path, filename)

            f = open(filename)
            _grammar = f.read()
            _grammar.replace("\n", "")
            f.close()
        # ...

        # ...
        self.grammar = _grammar
        # ...

        # ...
        if classes is None:
            self.model = metamodel_from_str(_grammar)
        else:
            self.model = metamodel_from_str(_grammar, classes=classes)
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
                   DeclarationStmt, \
                   AssignStmt, \
                   DelStmt, \
                   PassStmt, \
                   FlowStmt, BreakStmt, ContinueStmt, \
                   RaiseStmt, YieldStmt, ReturnStmt, \
                   IfStmt, ForStmt, FunctionDefStmt, \
                   ImportFromStmt, \
                   # numpy statments
                   NumpyZerosStmt, NumpyZerosLikeStmt, \
                   NumpyOnesStmt, NumpyLinspaceStmt
                   ]

        try:
            filename = kwargs["filename"]
        except:
            filename = "grammar.tx"

        super(PyccelParser, self).__init__(filename = filename, \
                                         classes=classes)

    def parse_from_file(self, filename):
        """Parse a set of instructions with respect to the grammar and returns
        the AST.

        filename: str
            a file containing the instructions to parse.
        """
        ast = super(PyccelParser, self).parse_from_file(filename)

        return ast
