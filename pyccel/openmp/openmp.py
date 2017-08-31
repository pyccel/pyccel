# coding: utf-8

"""
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.syntax import BasicStmt

class Openmp(object):
    """Class for Openmp syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Openmp.

        """
        self.statements = kwargs.pop('statements', [])

class OpenmpStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.stmt = kwargs.pop('stmt')

        super(OpenmpStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> OpenmpStmt: expr")
        pass

class ParallelStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(ParallelStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> ParallelStmt: expr")
        pass

class LoopStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(LoopStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> LoopStmt: expr")
        pass

class ParallelNumThreadClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.thread = kwargs.pop('thread')

        super(ParallelNumThreadClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> ParallelNumThreadClause: expr")
        pass

class ParallelDefaultClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(ParallelDefaultClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> ParallelDefaultClause: expr")
        pass

class ParallelProcBindClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(ParallelProcBindClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> ParallelProcBindClause: expr")
        pass

class PrivateClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(PrivateClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> PrivateClause: expr")
        pass

class SharedClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(SharedClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> SharedClause: expr")
        pass

class FirstPrivateClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(FirstPrivateClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> FirstPrivateClause: expr")
        pass

class LastPrivateClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(LastPrivateClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> LastPrivateClause: expr")
        pass

class CopyinClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(CopyinClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> CopyinClause: expr")
        pass

class ReductionClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.op   = kwargs.pop('op')
        self.args = kwargs.pop('args')

        super(ReductionClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> ReductionClause: expr")
        pass

class CollapseClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(CollapseClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> CollapseClause: expr")
        pass

class OrderedClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n', None)

        super(OrderedClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> OrderedClause: expr")
        pass

class ScheduleClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.kind       = kwargs.pop('kind')
        self.chunk_size = kwargs.pop('chunk_size', None)

        super(ScheduleClause, self).__init__(**kwargs)

    @property
    def expr(self):
        print("> ScheduleClause: expr")
        pass







def parse(filename, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'openmp.tx')
    classes = [Openmp, OpenmpStmt, \
               ParallelStmt, \
               LoopStmt, \
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
               ScheduleClause, \
               OrderedClause \
              ]
    meta = metamodel_from_file(grammar, debug=debug, classes=classes)

    # Instantiate model
    model = meta.model_from_file(filename)

    d = {}
    for stmt in model.statements:
        if isinstance(stmt, OpenmpStmt):
            print stmt.stmt
#            module = str(stmt.dotted_name.names[0])
#            names  = [str(n) for n in stmt.import_as_names.names]
#            d[module] = names

    return d

####################################
if __name__ == '__main__':
    filename = 'test.py'
    parse(filename, debug=False)
