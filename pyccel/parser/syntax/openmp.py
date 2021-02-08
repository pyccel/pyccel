# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
"""

from os.path import join, dirname

from textx.metamodel import metamodel_from_file

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import OmpAnnotatedComment, OMP_For_Loop, OMP_Parallel_Construct, OMP_Single_Construct,\
        Omp_End_Clause, OMP_Critical_Construct, OMP_Master_Construct,\
        OMP_Masked_Construct, OMP_Task_Construct, OMP_Cancel_Construct, OMP_Target_Construct, OMP_Teams_Construct, OMP_Sections_Construct, OMP_Section_Construct

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

        super().__init__(**kwargs)

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
        elif isinstance(stmt, OmpTaskyieldConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpTaskConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpFlushConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpCancelConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpTargetConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpTeamsConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpDistributeConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpSectionConstruct):
            return stmt.expr
        elif isinstance(stmt, OmpSectionsConstruct):
            return stmt.expr
        else:
            raise TypeError('Wrong stmt for OpenmpStmt')

class OmpParallelConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses  = kwargs.pop('clauses')
        self.combined = kwargs.pop('combined')

        super().__init__(**kwargs)

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
        combined = None
        if isinstance(self.combined, OmpForSimd):
            _valid_clauses = _valid_clauses + _valid_loop_clauses
            if 'simd' in self.combined.expr:
                _valid_clauses = _valid_clauses + _valid_simd_clauses
            combined = self.combined.expr
        if isinstance(self.combined, OmpMaskedTaskloop):
            _valid_clauses = _valid_clauses + (OmpFilter,)
            if 'simd' in self.combined.expr:
                _valid_clauses = _valid_clauses + _valid_simd_clauses
            if 'taskloop' in self.combined.expr:
                _valid_clauses = _valid_clauses + _valid_taskloop_clauses
            combined = self.combined.expr
        if isinstance(self.combined, OmpPSections):
            _valid_clauses = _valid_clauses + _valid_sections_clauses
            combined = self.combined.expr
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpParallelConstruct')

        return OMP_Parallel_Construct(txt, combined)

class OmpLoopConstruct(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpLoopConstruct: expr")

        txt = ''
        for clause in self.clauses:
            if isinstance(clause, _valid_loop_clauses):
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
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskLoopConstruct: expr")

        _valid_clauses = _valid_taskloop_clauses + (OmpinReduction,)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpTaskLoopConstruct. Given : ', \
                                type(clause))
        return OmpAnnotatedComment(txt)

class OmpTaskConstruct(BasicStmt):
    """Class representing a Task Construct """
    def __init__(self, **kwargs):
        """
        """
        self.clauses  = kwargs.pop('clauses')
        self.name     = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskConstruct: expr")

        _valid_clauses = (OmpPriority, \
                          OmpDefault, \
                          OmpPrivate, \
                          OmpShared, \
                          OmpFirstPrivate, \
                          OmpUntied, \
                          OmpMergeable, \
                          OmpinReduction, \
                          OmpDepend)

        txt = self.name
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
        self.name    = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSingleConstruct: expr")

        _valid_clauses = (OmpPrivate, \
                         OmpFirstPrivate)

        txt = self.name
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
        self.name    = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCriticalConstruct: expr")

        _valid_clauses = (OmpCriticalName)

        txt = self.name
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

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSimdConstruct: expr")

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_simd_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpSimdConstruct')

        return OmpAnnotatedComment(txt)

class OmpMasterConstruct(BasicStmt):
    """Class representing the master construct."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpMasterConstruct: expr")

        txt = self.name

        return OMP_Master_Construct(txt)

class OmpMaskedConstruct(BasicStmt):
    """Class representing the Masked construct."""
    def __init__(self, **kwargs):
        self.name     = kwargs.pop('name')
        self.combined = kwargs.pop('combined', None)
        self.clauses  = kwargs.pop('clauses')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpMaskedConstruct: expr")

        _valid_clauses = (OmpFilter,)

        combined = None
        if isinstance(self.combined, OmpTaskloopSimd):
            combined = self.combined.expr
            if 'simd' in self.combined.expr:
                _valid_clauses = _valid_clauses + _valid_simd_clauses
            _valid_clauses = _valid_clauses + _valid_taskloop_clauses
        txt = ''
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpMaskedConstruct')

        return OMP_Masked_Construct(txt, combined)

class OmpSectionsConstruct(BasicStmt):
    """Class representing a Sections stmt."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.clauses = kwargs.pop('clauses')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSectionsConstruct: expr")

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_sections_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpSectionsConstruct')

        return OMP_Sections_Construct(txt)

class OmpSectionConstruct(BasicStmt):
    """Class representing a Section stmt."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpSectionConstruct: expr")

        txt = self.name
        return OMP_Section_Construct(txt)

class OmpDistributeConstruct(BasicStmt):
    """Class representing a Barrier stmt."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.clauses = kwargs.pop('clauses')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpDistributeConstruct: expr")

        _valid_clauses = (OmpPrivate, \
                          OmpFirstPrivate, \
                          OmpLastPrivate, \
                          OmpCollapse)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpDistributeConstruct')

        return OmpAnnotatedComment(txt)

class OmpBarrierConstruct(BasicStmt):
    """Class representing a Barrier stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpBarrierConstruct: expr")

        txt = self.name
        return OmpAnnotatedComment(txt)

class OmpTaskWaitConstruct(BasicStmt):
    """Class representing a TaskWait stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskWaitConstruct: expr")

        txt = self.name
        return OmpAnnotatedComment(txt)

class OmpTaskyieldConstruct(BasicStmt):
    """Class representing a Taskyield stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTaskyieldConstruct: expr")

        txt = self.name
        return OmpAnnotatedComment(txt)

class OmpFlushConstruct(BasicStmt):
    """Class representing a Flush stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses  = kwargs.pop('clauses')
        self.name     = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpFlushConstruct: expr")

        _valid_clauses = (FlushList)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0}{1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpFlushConstruct')
        return OmpAnnotatedComment(txt)

class OmpCancelConstruct(BasicStmt):
    """Class representing a Cancel stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses  = kwargs.pop('clauses')
        self.name = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCancelConstruct: expr")

        _valid_clauses = (OmpCancelType)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpCancelConstruct')
        return OMP_Cancel_Construct(txt)

class OmpTargetConstruct(BasicStmt):
    """Class representing a Target stmt."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses  = kwargs.pop('clauses')
        self.name     = kwargs.pop('name')
        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTargetConstruct: expr")

        _valid_clauses = (OmpPrivate,\
                          OmpLastPrivate, \
                          OmpinReduction, \
                          OmpDepend, \
                          OmpMap)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpCancelConstruct')
        return OMP_Target_Construct(txt)

class OmpTeamsConstruct(BasicStmt):
    """Class representing a Teams stmt ."""
    def __init__(self, **kwargs):
        """
        """
        self.name     = kwargs.pop('name')
        self.clauses  = kwargs.pop('clauses')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpTeamsConstruct: expr")

        _valid_clauses = (OmpPrivate,\
                          OmpLastPrivate, \
                          OmpShared, \
                          OmpReduction, \
                          OmpNumTeams, \
                          OmpThreadLimit)

        txt = self.name
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Wrong clause for OmpTeamsConstruct')
        return OMP_Teams_Construct(txt)

class OmpAtomicConstruct(BasicStmt):
    """Class representing an Atomic stmt ."""
    def __init__(self, **kwargs):
        """
        """
        self.name     = kwargs.pop('name')
        self.clauses  = kwargs.pop('clauses')

        super().__init__(**kwargs)

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
        return OmpAnnotatedComment(txt)

class OmpEndClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.construct = kwargs.pop('construct')
        self.simd      = kwargs.pop('simd', '')
        self.nowait    = kwargs.pop('nowait', '')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpEndClause: expr")

        construct = ' '.join(self.construct)
        txt = 'end {0} {1} {2}'.format(construct, self.simd, self.nowait)
        return Omp_End_Clause(txt)

class OmpNumThread(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.thread = kwargs.pop('thread')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> OmpNumThread: expr")

        thread = self.thread
        return 'num_threads({})'.format(thread)

class OmpNumTeams(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.teams = kwargs.pop('teams')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> OmpNumTeams: expr")

        teams = self.teams
        return 'num_teams({})'.format(teams)

class OmpThreadLimit(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.limit = kwargs.pop('limit')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> OmpThreadLimit: expr")

        limit = self.limit
        return 'thread_limit({})'.format(limit)

class OmpNumTasks(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.tasks = kwargs.pop('tasks')

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpPrivate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'private({})'.format(args)

class FlushList(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Flush: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return '{}({})'.format(self.name, args)

class OmpCriticalName(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCopyin: expr")

        args = ', '.join(str(arg) for arg in self.args)
        return 'copyin({})'.format(args)

class OmpReduction(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.op   = kwargs.pop('op')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpReduction: expr")

        op   = self.op
        args = ', '.join(str(arg) for arg in self.args)
        return 'reduction({0}: {1})'.format(op, args)

class OmpDepend(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.dtype   = kwargs.pop('dtype')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpDepend: expr")

        dtype   = self.dtype
        args = ', '.join(str(arg) for arg in self.args)
        return 'depend({0}: {1})'.format(dtype, args)

class OmpMap(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.mtype   = kwargs.pop('mtype')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpMap: expr")

        mtype   = self.mtype
        args = ', '.join(str(arg) for arg in self.args)
        return 'map({0} {1})'.format(mtype, args)

class OmpinReduction(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.ctype  = kwargs.pop('ctype')
        self.op     = kwargs.pop('op')
        self.args   = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpinReduction: expr")

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpAtomicClause: expr")

        return '{}'.format(self.name)

class OmpCancelType(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpCancelType: expr")

        return '{}'.format(self.name)

class AtomicMemoryClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AtomicMemoryClause: expr")

        return '{}'.format(self.name)

class OmpForSimd(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.fname = kwargs.pop('fname')
        self.sname = kwargs.pop('sname', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined For Simd: expr")

        txt = self.fname
        if self.sname:
            txt = txt + ' ' + self.sname
        return '{}'.format(txt)

class OmpMaskedTaskloop(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.mname = kwargs.pop('mname')
        self.tname = kwargs.pop('tname', None)
        self.sname = kwargs.pop('sname', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Masked Taskloop: expr")

        txt = self.mname
        if self.tname:
            txt = txt + ' ' + self.tname
            if self.sname:
                txt = txt + ' ' + self.sname
        return '{}'.format(txt)

class OmpPSections(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.sname = kwargs.pop('sname')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Sections: expr")

        txt = self.sname
        return txt

class OmpTaskloopSimd(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.tname = kwargs.pop('tname')
        self.sname = kwargs.pop('sname', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Taskloop Simd: expr")

        txt = self.tname
        if self.sname:
            txt += ' ' + self.sname
        return txt

#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.

_valid_sections_clauses = (OmpPrivate, \
                           OmpFirstPrivate, \
                           OmpLastPrivate, \
                           OmpReduction)

_valid_simd_clauses = (OmpLinear, \
                       OmpReduction, \
                       OmpCollapse, \
                       OmpLastPrivate)

_valid_taskloop_clauses = (OmpShared, \
                           OmpPrivate, \
                           OmpFirstPrivate, \
                           OmpLastPrivate, \
                           OmpReduction, \
                           OmpinReduction, \
                           OmpNumTasks, \
                           OmpGrainSize, \
                           OmpCollapse, \
                           OmpUntied, \
                           OmpMergeable, \
                           OmpNogroup, \
                           OmpPriority)

_valid_loop_clauses = (OmpPrivate, \
                       OmpFirstPrivate, \
                       OmpLastPrivate, \
                       OmpReduction, \
                       OmpSchedule, \
                       OmpCollapse, \
                       OmpLinear, \
                       OmpOrdered)

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
                  OmpTaskWaitConstruct,
                  OmpTaskyieldConstruct,
                  OmpTaskConstruct,
                  OmpFlushConstruct,
                  OmpCancelConstruct,
                  OmpTargetConstruct,
                  OmpTeamsConstruct,
                  OmpDistributeConstruct,
                  OmpSectionsConstruct,
                  OmpSectionConstruct]

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
               OmpinReduction,
               OmpNumTasks,
               OmpGrainSize,
               OmpUntied,
               OmpMergeable,
               OmpNogroup,
               OmpPriority,
               OmpAtomicClause,
               AtomicMemoryClause,
               OmpDepend,
               FlushList,
               OmpCancelType,
               OmpMap,
               OmpNumTeams,
               OmpThreadLimit,
               OmpForSimd,
               OmpMaskedTaskloop,
               OmpPSections,
               OmpTaskloopSimd]

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
