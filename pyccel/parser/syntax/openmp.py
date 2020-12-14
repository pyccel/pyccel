# coding: utf-8

"""
"""

from os.path import join, dirname

from textx.metamodel import metamodel_from_file

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import OMP_For_Loop, OMP_Parallel_Construct, OMP_Single_Construct,\
        Omp_End_Clause, OMP_Critical_Construct, OMP_Barrier_Construct, OMP_Master_Construct,\
        OMP_Masked_Construct, OMP_TaskLoop_Construct, OMP_Simd_Construct, OMP_Atomic_Construct, OMP_TaskWait_Construct, OMP_Task_Construct

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
        elif isinstance(stmt, OmpMasterConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpMaskedConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpTaskLoopConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpSimdConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpAtomicConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpTaskWaitConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpTaskConstruct):
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

class OmpTaskLoopConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(OmpTaskLoopConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskLoopConstruct: expr")

        _valid_clauses = (OmpShared, \
                         OmpPrivate, \
                         OmpFirstPrivate, \
                         OmpLastPrivate, \
                         OmpTaskloopReduction, \
                         OmpNumTasks, \
                         OmpGrainSize, \
                         OmpCollapse, \
                         OmpUntied, \
                         OmpMergeable, \
                         OmpNogroup, \
                         OmpPriority)

        txt = ''
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpTaskLoopConstruct. Given : ', \
                                type(clause))
        return OMP_TaskLoop_Construct(txt)

class OmpTaskConstruct(BasicStmt):
    """Class representing a Task Construct """
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(OmpTaskConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskConstruct: expr")

        _valid_clauses = ()
        
        txt = 'task'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpTaskConstruct')

        return OMP_Task_Construct(txt)

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

class OmpSimdConstruct(BasicStmt):
    """Class representing a Simd construct."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.clauses = kwargs.pop('clauses')

        super(OmpSimdConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSimdConstruct: expr")

        _valid_clauses = (OmpLinear, \
                         OmpReduction, \
                         OmpCollapse, \
                         OmpLastPrivate)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
              raise TypeError('Wrong clause for OmpSimdConstruct')

        return OMP_Simd_Construct(txt)

class OmpMasterConstruct(BasicStmt):
    """Class representing the master construct."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        super(OmpMasterConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpMasterConstruct: expr")

        txt = self.name

        return OMP_Master_Construct(txt)

class OmpMaskedConstruct(BasicStmt):
    """Class representing the Masked construct."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.clauses = kwargs.pop('clauses')

        super(OmpMaskedConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpMaskedConstruct: expr")

        _valid_clauses = (OmpFilter)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpMaskedConstruct')

        return OMP_Masked_Construct(txt)

class OmpBarrierConstruct(BasicStmt):
    """Class representing a Barrier stmt."""
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

class OmpTaskWaitConstruct(BasicStmt):
    """Class representing a TaskWait stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        super(OmpTaskWaitConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskWaitConstruct: expr")

        txt = self.name
        return OMP_TaskWait_Construct(txt)

class OmpAtomicConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name   = kwargs.pop('name')
        self.clauses = kwargs.pop('clauses')

        super(OmpAtomicConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpBarrierConstruct: expr")

        _valid_clauses = (OmpAtomicClause, \
                          AtomicMemoryClause)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpAtomicConstruct')
        return OMP_Atomic_Construct(txt)

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

class OmpNumTasks(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.tasks = kwargs.pop('tasks')

        super(OmpNumTasks, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> OmpNumTasks: expr")

        tasks = self.tasks
        return 'num_tasks({})'.format(tasks)

class OmpGrainSize(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.size = kwargs.pop('tasks')

        super(OmpGrainSize, self).__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> OmpGrainSize: expr")

        size = self.size
        return 'grainsize({})'.format(size)

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

class OmpTaskloopReduction(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.ctype  = kwargs.pop('ctype') 
        self.op     = kwargs.pop('op')
        self.args   = kwargs.pop('args')

        super(OmpTaskloopReduction, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskloopReduction: expr")

        # TODO check if variable exist in namespace
        ctype = self.ctype
        op    = self.op
        args  = ', '.join(str(arg) for arg in self.args)
        return '{0}({1}: {2})'.format(ctype, op, args)

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

class OmpFilter(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.n = kwargs.pop('n')

        super(OmpFilter, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpFilter: expr")

        return '{}({})'.format(self.name, self.n)

class OmpUntied(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super(OmpUntied, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpUntied: expr")

        return 'untied'

class OmpMergeable(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super(OmpMergeable, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpMergeable: expr")

        return 'mergeable'

class OmpNogroup(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super(OmpNogroup, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpNogroup: expr")

        return 'nogroup'

class OmpPriority(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.n = kwargs.pop('n')

        super(OmpPriority, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpPriority: expr")

        return '{}({})'.format(self.name, self.n)

class OmpAtomicClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super(OmpAtomicClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpAtomicClause: expr")

        return '{}'.format(self.name)

class AtomicMemoryClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super(AtomicMemoryClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AtomicMemoryClause: expr")

        return '{}'.format(self.name)

#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
omp_directives = [OmpParallelConstruct,
                  OmpLoopConstruct,
                  OmpSingleConstruct,
                  OmpEndClause,
                  OmpCriticalConstruct,
                  OmpBarrierConstruct,
                  OmpMasterConstruct,
                  OmpMaskedConstruct,
                  OmpTaskLoopConstruct,
                  OmpSimdConstruct,
                  OmpAtomicConstruct,
                  OmpTaskWaitConstruct]

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
               OmpCriticalName,
               OmpFilter,
               OmpTaskloopReduction,
               OmpNumTasks,
               OmpGrainSize,
               OmpUntied,
               OmpMergeable,
               OmpNogroup,
               OmpPriority,
               OmpAtomicClause,
               AtomicMemoryClause]

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
