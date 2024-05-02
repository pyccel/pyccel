# coding: utf-8

"""
This module contains the syntax associated to the types.tx grammar
"""

from os.path import join, dirname

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import Variable
from pyccel.ast.datatypes import datatype, VariableType, FunctionType


def _construct_dtype(dtype):
    """."""
    if isinstance(dtype, FunctionTypeStmt):
        return dtype.expr
    else:
        return datatype(str(dtype))


class HiMi(object):
    """Class for HiMi syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for HiMi.

        """
        self.statements = kwargs.pop('statements', [])


class DeclareTypeStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.dtype = kwargs.pop('dtype')

        super(DeclareTypeStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        name = str(self.name)
        dtype = _construct_dtype(self.dtype)

        return VariableType(dtype, name)


class DeclareVariableStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.dtype = kwargs.pop('dtype')

        super(DeclareVariableStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        name = str(self.name)
        dtype = datatype(str(self.dtype))
        return Variable(dtype, name)


class DeclareFunctionStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """

        """
        self.name = kwargs.pop('name')
        self.dtype = kwargs.pop('dtype')

        super(DeclareFunctionStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        name = str(self.name)
        dtype = _construct_dtype(self.dtype)

        # TODO must return a TypedFunction
        return Variable(dtype, name)


class FunctionTypeStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """

        """
        self.domains = kwargs.pop('domains')

        super(FunctionTypeStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        domains = []
        for d in self.domains:
            domains.append(datatype(str(d)))
        return FunctionType(domains)



#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
types_classes = [HiMi,
                 FunctionTypeStmt,
                 DeclareTypeStmt,
                 DeclareFunctionStmt,
                 DeclareVariableStmt]

def parse(filename=None, stmts=None, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, '../grammar/himi.tx')

    from textx.metamodel import metamodel_from_file
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
#    print (parse(stmts='T = int'))
#    print (parse(stmts='x : int'))
#    print (parse(stmts='f :: int -> double'))
#    print (parse(stmts='T = int -> double'))
#    print (parse(stmts='T = int -> double -> double'))
#    print (parse(stmts='int -> double')) # TODO to be removed. only for testing
    print (parse(stmts='int -> double -> double')) # TODO to be removed. only for testing
