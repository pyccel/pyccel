# coding: utf-8

"""
This module contains the syntax associated to the types.tx grammar
"""

from os.path import join, dirname
from sympy.utilities.iterables import iterable
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast import DataType, datatype, AliasDataType
from pyccel.ast import Variable

DEBUG = False

class HiMi(object):
    """Class for HiMi syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for HiMi.

        """
        self.statements = kwargs.pop('statements', [])


class DeclareTypeStmt(BasicStmt):
    """Base class representing a type declaration in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a DeclareTypeStmt statement.

        name: str
            type name
        dtype: str
            datatype
        """
        self.name = kwargs.pop('name')
        self.dtype = kwargs.pop('dtype')

        super(DeclareTypeStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = datatype(str(self.dtype))
        name = str(self.name)
        return AliasDataType(name, dtype)


class DeclareVariableStmt(BasicStmt):
    """Base class representing a variable declaration in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a DeclareTypeStmt statement.

        name: str
            type name
        dtype: str
            datatype
        """
        self.name = kwargs.pop('name')
        self.dtype = kwargs.pop('dtype')

        super(DeclareVariableStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = datatype(str(self.dtype))
        name = str(self.name)
        return Variable(dtype, name)



#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
types_classes = [HiMi,
                 DeclareTypeStmt,
                 DeclareVariableStmt]

def parse(filename=None, stmts=None, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, '../grammar/himi.tx')

    meta = metamodel_from_file(grammar, debug=debug, classes=types_classes)

    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')

    stmts = []
    for stmt in model.statements:
        e = stmt.expr
        stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts

######################
if __name__ == '__main__':
    print(parse(stmts='E = int'))
    print(parse(stmts='x :: int'))
