# coding: utf-8

"""
This module contains the syntax associated to the types.tx grammar
"""

from os.path import join, dirname
from sympy.utilities.iterables import iterable
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast import DataType, datatype
from pyccel.ast import Variable


# ...
class VariableType(DataType):

    def __init__(self, rhs, alias):
        self._alias = alias
        self._rhs = rhs
        self._name = rhs._name

    @property
    def alias(self):
        return self._alias

class FunctionType(DataType):

    def __init__(self, domain, codomain):
        self._domain = domain
        self._codomain = codomain
        self._name = '{V} -> {W}'.format(V=domain, W=codomain)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain
# ...


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
        dtype = self.dtype
        dtype = datatype(dtype)
        name = str(self.name)
        return VariableType(dtype, name)


class DeclareVariableStmt(BasicStmt):
    """."""

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
        name = str(self.name)
        dtype = datatype(str(self.dtype))
        return Variable(dtype, name)


class FunctionTypeStmt(BasicStmt):
    """."""

    def __init__(self, **kwargs):
        """
        Constructor for a DeclareTypeStmt statement.

        """
        self.domain = kwargs.pop('domain')
        self.codomain = kwargs.pop('codomain')

        super(FunctionTypeStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        domain = datatype(str(self.domain))
        codomain = datatype(str(self.codomain))
        return FunctionType(domain, codomain)



#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
types_classes = [HiMi,
                 FunctionTypeStmt,
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
#    print (parse(stmts='T = int'))
#    print (parse(stmts='x : int'))
#    print (parse(stmts='f :: int -> double'))
#    print (parse(stmts='T = int -> double'))
    print (parse(stmts='int -> double'))
