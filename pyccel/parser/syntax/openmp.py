# coding: utf-8

"""
"""

from os.path import join, dirname

from textx.metamodel import metamodel_from_file

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import OMP_For_Loop, OMP_Parallel_Construct, OMP_Single_Construct, Omp_End_Clause, OMP_Critical_Construct, OMP_Barrier_Construct

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
        if isinstance(stmt, OmpEndClause):
            return stmt.expr
        elif isinstance(stmt, OmpParallelConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpLoopConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpSingleConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpCriticalConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpBarrierConstruct):
            return stmt.expr
        else:
            raise TypeError('Wrong stmt for OpenmpStmt')

class OmpParallelConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')
        self.combined = kwargs.pop('combined')

        super(OmpParallelConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpParallelConstruct: expr")

        _valid_clauses = (OmpNumThread, \
                         OmpDefault, \
                         OmpPrivate, \
                         OmpShared, \
                         OmpFirstPrivate, \
                         OmpCopyin, \
                         OmpReduction, \
                         OmpProcBind)

        txt = ''
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpParallelConstruct')

        return OMP_Parallel_Construct(txt, self.combined)

class OmpLoopConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(OmpLoopConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpLoopConstruct: expr")

        _valid_clauses = (OmpPrivate, \
                         OmpFirstPrivate, \
                         OmpLastPrivate, \
                         OmpReduction, \
                         OmpSchedule, \
                         OmpCollapse, \
                         OmpLinear, \
                         OmpOrdered)

        txt = ''
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpLoopConstruct. Given : ', \
                                type(clause))
        return OMP_For_Loop(txt)

class OmpSingleConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(OmpSingleConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSingleConstruct: expr")

        _valid_clauses = (OmpPrivate, \
                         OmpFirstPrivate)

        txt = 'single'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpSingleConstruct')

        return OMP_Single_Construct(txt)

class OmpCriticalConstruct(BasicStmt):
    """Class representing a Critical stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(OmpCriticalConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCriticalConstruct: expr")

        _valid_clauses = (OmpCriticalName)

        txt = 'critical'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
              raise TypeError('Wrong clause for OmpCriticalConstruct')

        return OMP_Critical_Construct(txt)

class OmpBarrierConstruct(BasicStmt):
    """Class representing a Critical stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        super(OmpBarrierConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpBarrierConstruct: expr")

        txt = self.name

        return OMP_Barrier_Construct(txt)

class OmpEndClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.construct = kwargs.pop('construct')
        self.simd      = kwargs.pop('simd', '')
        self.nowait    = kwargs.pop('nowait', '')

        super(OmpEndClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpEndClause: expr")

        txt = 'end {0} {1} {2}'.format(self.construct, self.simd, self.nowait)
        return Omp_End_Clause(txt)

class OmpNumThread(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.thread = kwargs.pop('thread')

        super(OmpNumThread, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> OmpNumThread: expr")

        thread = self.thread
        return 'num_threads({})'.format(thread)

class OmpDefault(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(OmpDefault, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpDefault: expr")

        return 'default({})'.format(self.status)

class OmpProcBind(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(OmpProcBind, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpProcBind: expr")

        return 'proc_bind({})'.format(self.status)

class OmpPrivate(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(OmpPrivate, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpPrivate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'private({})'.format(args)

class OmpCriticalName(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(OmpCriticalName, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCriticalName: expr")

        # TODO check if variable exist in namespace
        txt = str(self.args)
        return '({})'.format(txt)

class OmpShared(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(OmpShared, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpShared: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'shared({})'.format(args)

class OmpFirstPrivate(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(OmpFirstPrivate, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpFirstPrivate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'firstprivate({})'.format(args)

class OmpLastPrivate(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(OmpLastPrivate, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpLastPrivate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'lastprivate({})'.format(args)

class OmpCopyin(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(OmpCopyin, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCopyin: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'copyin({})'.format(args)

class OmpReduction(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.op   = kwargs.pop('op')
        self.args = kwargs.pop('args')

        super(OmpReduction, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpReduction: expr")

        # TODO check if variable exist in namespace
        op   = self.op
        args = ', '.join(str(arg) for arg in self.args)
        return 'reduction({0}: {1})'.format(op, args)

class OmpCollapse(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(OmpCollapse, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCollapse: expr")

        return 'collapse({})'.format(self.n)

class OmpOrdered(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n', None)

        super(OmpOrdered, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpOrdered: expr")

        if self.n:
            return 'ordered({})'.format(self.n)
        else:
            return 'ordered'

class OmpLinear(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.val  = kwargs.pop('val')
        self.step = kwargs.pop('step')

        super(OmpLinear, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpLinear: expr")

        return 'linear({0}:{1})'.format(self.val, self.step)

class OmpSchedule(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.kind       = kwargs.pop('kind')
        self.chunk_size = kwargs.pop('chunk_size', None)

        super(OmpSchedule, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSchedule: expr")

        if self.chunk_size:
            return 'schedule({0}, {1})'.format(self.kind, self.chunk_size)
        else:
            return 'schedule({0})'.format(self.kind)
#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
omp_directives = [OmpParallelConstruct,
                  OmpLoopConstruct,
                  OmpSingleConstruct,
                  OmpEndClause,
                  OmpCriticalConstruct,
                  OmpBarrierConstruct]

omp_clauses = [OmpCollapse,
               OmpCopyin,
               OmpFirstPrivate,
               OmpLastPrivate,
               OmpLinear,
               OmpOrdered,
               OmpNumThread,
               OmpDefault,
               OmpPrivate,
               OmpProcBind,
               OmpPrivate,
               OmpReduction,
               OmpSchedule,
               OmpShared,
               OmpCriticalName]

omp_classes = [Openmp, OpenmpStmt] + omp_directives + omp_clauses


this_folder = dirname(__file__)

# Get meta-model from language description
grammar = join(this_folder, '../grammar/openmp.tx')

meta = metamodel_from_file(grammar, classes=omp_classes)

def parse(filename=None, stmts=None):
    """ Parse openmp pragmas

      Parameters
      ----------

      filename : str

      stmts : list

      Results
      -------

      stmts : list

    """
    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')

    stmts = []
    for stmt in model.statements:
        if isinstance(stmt, OpenmpStmt):
            e = stmt.stmt.expr
            stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
if __name__ == '__main__':
    print(parse(stmts='#$omp parallel'))
    print(parse(stmts='#$omp do private ( ipart, pos, spana, lefta, righta, valuesa, spanb, leftb, rightb, valuesb, E)'))
    print(parse(stmts='#$omp do private(ipart, pos, spana, lefta, righta, valuesa, spanb, leftb, rightb, valuesb,E, B)'))
    print(parse(stmts='#$omp end do'))
    print(parse(stmts='#$omp end parallel'))
