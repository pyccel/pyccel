# coding: utf-8

"""
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.syntax import BasicStmt
from pyccel.types.ast import AnnotatedComment

DEBUG = False

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
        if DEBUG:
            print("> OpenmpStmt: expr")

        stmt = self.stmt
        if isinstance(stmt, EndConstructClause):
            return stmt.expr
        elif isinstance(stmt, ParallelStmt):
            return stmt.expr
        elif isinstance(stmt, LoopStmt):
            return stmt.expr
        elif isinstance(stmt, SingleStmt):
            return stmt.expr
        else:
            raise TypeError('Wrong stmt for OpenmpStmt')

class ParallelStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(ParallelStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> ParallelStmt: expr")

        valid_clauses = (ParallelNumThreadClause, \
                         ParallelDefaultClause, \
                         PrivateClause, \
                         SharedClause, \
                         FirstPrivateClause, \
                         CopyinClause, \
                         ReductionClause, \
                         ParallelProcBindClause)

        txt = 'parallel'
        for clause in self.clauses:
            if isinstance(clause, valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for ParallelStmt')

        return AnnotatedComment('omp', txt)

class LoopStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(LoopStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> LoopStmt: expr")

        valid_clauses = (PrivateClause, \
                         FirstPrivateClause, \
                         LastPrivateClause, \
                         ReductionClause, \
                         ScheduleClause, \
                         CollapseClause, \
                         LinearClause, \
                         OrderedClause)

        txt = 'do'
        for clause in self.clauses:
            if isinstance(clause, valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for LoopStmt')

        return AnnotatedComment('omp', txt)

class SingleStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(SingleStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> SingleStmt: expr")

        valid_clauses = (PrivateClause, \
                         FirstPrivateClause)

        txt = 'single'
        for clause in self.clauses:
            if isinstance(clause, valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for SingleStmt')

        return AnnotatedComment('omp', txt)

class EndConstructClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.construct = kwargs.pop('construct')
        self.simd      = kwargs.pop('simd', '')
        self.nowait    = kwargs.pop('nowait', '')

        super(EndConstructClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> EndConstructClause: expr")

        txt = 'end {0} {1} {2}'.format(self.construct, self.simd, self.nowait)
        return AnnotatedComment('omp', txt)

class ParallelNumThreadClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.thread = kwargs.pop('thread')

        super(ParallelNumThreadClause, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> ParallelNumThreadClause: expr")

        thread = self.thread
        return 'num_threads({})'.format(thread)

class ParallelDefaultClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(ParallelDefaultClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> ParallelDefaultClause: expr")

        return 'default({})'.format(self.status)

class ParallelProcBindClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(ParallelProcBindClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> ParallelProcBindClause: expr")

        return 'proc_bind({})'.format(self.status)

class PrivateClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(PrivateClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> PrivateClause: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'private({})'.format(args)

class SharedClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(SharedClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> SharedClause: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'shared({})'.format(args)

class FirstPrivateClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(FirstPrivateClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> FirstPrivateClause: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'firstprivate({})'.format(args)

class LastPrivateClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(LastPrivateClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> LastPrivateClause: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'lastprivate({})'.format(args)

class CopyinClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(CopyinClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> CopyinClause: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'copyin({})'.format(args)

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
        if DEBUG:
            print("> ReductionClause: expr")

        # TODO check if variable exist in namespace
        op   = self.op
        args = ', '.join(str(arg) for arg in self.args)
        return 'copyin({0}: {1})'.format(op, args)

class CollapseClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(CollapseClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> CollapseClause: expr")

        return 'collapse({})'.format(self.n)

class OrderedClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n', None)

        super(OrderedClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OrderedClause: expr")

        if self.n:
            return 'ordered({})'.format(self.n)
        else:
            return 'ordered'

class LinearClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.val  = kwargs.pop('val')
        self.step = kwargs.pop('step')

        super(LinearClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> LinearClause: expr")

        return 'linear({0} : {1})'.format(self.val, self.step)


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
        if DEBUG:
            print("> ScheduleClause: expr")

        if self.chunk_size:
            return 'schedule({0}, {1})'.format(self.kind, self.chunk_size)
        else:
            return 'schedule({0})'.format(self.kind)




def parse(filename, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')
    classes = [Openmp, OpenmpStmt, \
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
               OrderedClause \
              ]
    meta = metamodel_from_file(grammar, debug=debug, classes=classes)

    # Instantiate model
    model = meta.model_from_file(filename)

    for stmt in model.statements:
        if isinstance(stmt, OpenmpStmt):
            e = stmt.stmt.expr
            print(e)

####################################
if __name__ == '__main__':
    filename = 'test.py'
    parse(filename, debug=False)
