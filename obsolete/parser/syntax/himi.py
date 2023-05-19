# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module contains the syntax associated to the types.tx grammar
"""

from os.path import join, dirname

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import Variable
from pyccel.ast.datatypes import datatype, VariableType, FunctionType

""" from pyccel.ast.datatypes #pylint: disable=pointless-string-statement

class VariableType(DataType):
    __slots__ = ('_alias','_rhs','_name')

    def __init__(self, rhs, alias):
        self._alias = alias
        self._rhs = rhs
        self._name = rhs._name

    @property
    def alias(self):
        return self._alias

class FunctionType(DataType):
￼    __slots__ = ('_domain','_codomain','_domains','_name')
￼
￼    def __init__(self, domains):
￼        self._domain = domains[0]
￼        self._codomain = domains[1:]
￼        self._domains = domains
￼        self._name = ' -> '.join('{}'.format(V) for V in self._domains)
￼
￼    @property
￼    def domain(self):
￼        return self._domain
￼
￼    @property
￼    def codomain(self):
￼        return self._codomain
"""


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
