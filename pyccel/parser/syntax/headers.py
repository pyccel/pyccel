# coding: utf-8

"""
"""

# TODO: - remove 'star' from everywhere

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import FunctionHeader, ClassHeader, MethodHeader, VariableHeader

DEBUG = False

class Header(object):
    """Class for Header syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Header.

        """
        self.statements = kwargs.pop('statements', [])

class TypeHeader(BasicStmt):
    """Base class representing a  header type in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: str
            variable type
        """
        self.dtype = kwargs.pop('dtype')
        self.trailer = kwargs.pop('trailer', [])

        super(TypeHeader, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dtype
        trailer = self.trailer
        if trailer:
            trailer = [str(i) for i in trailer.args]
        else:
            trailer = []
        return dtype, trailer
class TypeHeader(BasicStmt):
    pass

class UnionTypeStmt(BasicStmt):
    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list fo str
        """
        self.dtypes = kwargs.pop('dtype')

        super(UnionTypeStmt, self).__init__(**kwargs)


class VariableHeaderStmt(BasicStmt):
    """Base class representing a header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a VariableHeader statement.
        In the case of builtin datatypes, we export a Variable

        name: str
            variable name
        dec: list, tuple
            list of argument types
        """
        self.name = kwargs.pop('name')
        self.dec  = kwargs.pop('dec')

        super(VariableHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype, trailer = self.dec.expr
        star = None

        return VariableHeader(self.name, (dtype, trailer, star))

class FunctionHeaderStmt(BasicStmt):
    """Base class representing a function header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a FunctionHeader statement

        name: str
            function name
        kind: str
            one among {function, procedure, method}
        decs: list, tuple
            list of argument types
        results: list, tuple
            list of output types
        """
        self.name = kwargs.pop('name')
        self.kind = kwargs.pop('kind', None)
        self.static = kwargs.pop('static', None)
        self.decs = kwargs.pop('decs')
        self.results = kwargs.pop('results', None)

        super(FunctionHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO: do we need dtypes and results to be attributs of the class?
        dtypes = []
        for dec in self.decs:
            if isinstance(dec,UnionTypeStmt):
                l = []
                for i in dec.dtypes:
                    l += [i.expr +('',)]
                dtypes +=[l]
            else:       
                dtypes +=[[dec.expr + ('',)]]                

        if self.kind is None:
            kind = 'function'
        else:
            kind = str(self.kind)

        is_static = False
        if self.static == 'static':
            is_static = True

        if kind == 'method':
            cls_instance = dtypes[0][0][0]
            dtypes = dtypes[1:] # remove the attribut
            kind = 'procedure'
            if self.results:
                kind = 'function'
            return MethodHeader((cls_instance, self.name), dtypes, [],kind=kind )
        else:
            return FunctionHeader(self.name,
                                  dtypes,
                                  results=[],
                                  kind=kind,
                                  is_static=is_static)

class ClassHeaderStmt(BasicStmt):
    """Base class representing a class header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Header statement

        name: str
            class name
        options: list, tuple
            list of class options
        """
        self.name    = kwargs.pop('name')
        self.options = kwargs.pop('options')

        super(ClassHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        options = [str(i) for i in self.options]
        return ClassHeader(self.name, options)
#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.

hdr_classes = [Header, TypeHeader,Type,ListType,UnionTypeStmt,
               FunctionHeaderStmt,
               ClassHeaderStmt,
               VariableHeaderStmt]

def parse(filename=None, stmts=None, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, '../grammar/headers.tx')

    meta = metamodel_from_file(grammar, debug=debug, classes=hdr_classes)

    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')

    stmts = []
    for stmt in model.statements:
        e = stmt.stmt.expr
        stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts

######################
if __name__ == '__main__':
    print(parse(stmts='#$ header variable x :: int'))
    print(parse(stmts='#$ header variable x float [:, :]'))
    print(parse(stmts='#$ header function f(float [:], int [:]) results(int)'))
    print(parse(stmts='#$ header class Square(public)'))
    print(parse(stmts='#$ header method translate(Point, [double], [int], int[:,:], double[:])'))
