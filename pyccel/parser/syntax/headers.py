# coding: utf-8

"""
"""

# TODO: - remove 'star' from everywhere

from os.path import join, dirname
from sympy.utilities.iterables import iterable
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import FunctionHeader, ClassHeader, MethodHeader, VariableHeader
from pyccel.ast.core import MetaVariable , UnionType

DEBUG = False

class Header(object):
    """Class for Header syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Header.

        """
        self.statements = kwargs.pop('statements', [])

class ListType(BasicStmt):
    """Base class representing a  ListType in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list fo str
        """
        self.dtype = kwargs.pop('dtype')

        super(ListType, self).__init__(**kwargs)

    @property
    def expr(self):
        dtypes = [str(i) for i in self.dtype]
        if not (all(dtypes[0]==i for i in dtypes)):
            raise TypeError('all element of the TypeList must have the same type')
        d_var = {}
        d_var['datatype'] = dtypes[0]
        d_var['rank'] = len(dtypes)
        d_var['is_pointer'] = len(dtypes)>0
        d_var['allocatable'] = False
        return d_var

class Type(BasicStmt):
    """Base class representing a header type in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Type.

        dtype: str
            variable type
        """
        self.dtype = kwargs.pop('dtype')
        self.trailer = kwargs.pop('trailer', [])

        super(Type, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dtype
        trailer = self.trailer
        if trailer:
            trailer = [str(i) for i in trailer.args]
        else:
            trailer = []
        d_var={}
        d_var['datatype']=dtype
        d_var['rank'] = len(trailer)
        d_var['allocatable'] = len(trailer)>0
        d_var['is_pointer'] = False
        return d_var

class TypeHeader(BasicStmt):
    pass

# TODO must add expr property
class UnionTypeStmt(BasicStmt):
    def __init__(self, **kwargs):
        """
        Constructor for a TypeHeader.

        dtype: list fo str
        """
        self.dtypes = kwargs.pop('dtype')

        super(UnionTypeStmt, self).__init__(**kwargs)


class HeaderResults(BasicStmt):
    """Base class representing a HeaderResults in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a HeaderResults.

        decs: list of TypeHeader
        """
        self.decs = kwargs.pop('decs')

        super(HeaderResults, self).__init__(**kwargs)

    @property
    def expr(self):
        decs = [i.expr for i in self.decs]
        return decs


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
        dtype = self.dec.expr
        star = None

        return VariableHeader(self.name, (dtype, star))

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
                    l += [i.expr]
                dtypes += [UnionType(l)]

        if self.kind is None:
            kind = 'function'
        else:
            kind = str(self.kind)

        is_static = False
        if self.static == 'static':
            is_static = True

        results = []
        if self.results:
            results = self.results.expr

        if kind == 'method':
            cls_instance = dtypes[0].args[0]['datatype']
            dtypes = dtypes[1:] # remove the attribut
            kind = 'procedure'
            if results:
                kind = 'function'
            return MethodHeader((cls_instance, self.name), dtypes, [],kind=kind )
        else:
            return FunctionHeader(self.name,
                                  dtypes,
                                  results=results,
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
hdr_classes = [Header, TypeHeader,
               Type, ListType, UnionTypeStmt,
               HeaderResults,
               FunctionHeaderStmt,
               ClassHeaderStmt,
               VariableHeaderStmt,
               MetavarHeaderStmt]

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
    print(parse(stmts='#$ header function f(float|int, int [:]) results(int)'))
    print(parse(stmts='#$ header class Square(public)'))
    print(parse(stmts='#$ header method translate(Point, [double], [int], int[:,:], double[:])'))
    print(parse(stmts="#$ header metavar module_name='mpi'"))
