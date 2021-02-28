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
from pyccel.ast.omp import (OmpAnnotatedComment, OMP_For_Loop, OMP_Parallel_Construct,
                            OMP_Single_Construct, Omp_End_Clause, OMP_Critical_Construct,
                            OMP_Master_Construct, OMP_Masked_Construct, OMP_Task_Construct,
                            OMP_Cancel_Construct, OMP_Target_Construct, OMP_Teams_Construct,
                            OMP_Sections_Construct, OMP_Section_Construct)

DEBUG = True

class OmpConstruct(BasicStmt):
    """Class representing all OpenMP constructs."""
    def __init__(self, omp_type, vclauses, **kwargs):
        name     = kwargs.pop('name', None)
        clauses  = kwargs.pop('clauses', None)
        combined = kwargs.pop('combined', None)

        _valid_clauses = vclauses

        com = None
        if combined:
            if 'for' in combined.expr:
                _valid_clauses += _valid_loop_clauses
            if 'simd' in combined.expr:
                _valid_clauses += _valid_simd_clauses
            if 'taskloop' in combined.expr:
                _valid_clauses += _valid_taskloop_clauses
            if 'masked' in combined.expr:
                _valid_clauses += (OmpFilter,)
            if 'sections' in combined.expr:
                _valid_clauses += _valid_sections_clauses
            if 'distribute' in combined.expr:
                _valid_clauses += _valid_Distribute_clauses
            if 'parallel' in combined.expr:
                _valid_clauses += _valid_parallel_clauses
            if 'teams' in combined.expr:
                _valid_clauses += _valid_teams_clauses
            com = combined.expr
       
        txt = ''
        if name:
            txt += name
        if clauses:
            txt += check_get_clauses(self, _valid_clauses, clauses, combined)
        
        if combined:
            self._expr = omp_type(txt, com)
        else:
            self._expr = omp_type(txt)

        super().__init__(**kwargs)

def check_get_clauses(name, valid_clauses, clauses, combined = None):
    """
    Function to check if the clauses are correct for a given Construct.
    """
    txt = ''
    for clause in clauses:
        if isinstance(clause, valid_clauses):
            if isinstance(clause, OmpCopyin) and isinstance(combined, OmpTargetParallel):
                 msg = "Wrong clause " + type(clause).__name__
                 raise TypeError(msg)
            txt = '{0} {1}'.format(txt, clause.expr)
        else:
            msg = "Wrong clause " + type(clause).__name__
            raise TypeError(msg)
    return txt


class Openmp(object):
    """Class for Openmp syntax."""
    def __init__(self, **kwargs):
        self.statements = kwargs.pop('statements', [])

class OpenmpStmt(BasicStmt):
    """Class representing an OpenMP statement."""
    def __init__(self, **kwargs):
        self.stmt = kwargs.pop('stmt')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        stmt = self.stmt
        if isinstance(stmt, omp_directives):
            return stmt.expr
        else:
            raise TypeError('Wrong stmt for OpenmpStmt')

class OmpParallelConstruct(OmpConstruct):
    """Class representing a parallel construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Parallel_Construct, _valid_parallel_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpLoopConstruct(OmpConstruct):
    """Class representing a For loop construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_For_Loop, _valid_loop_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpTaskLoopConstruct(OmpConstruct):
    """Class representing a taskloop construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, (_valid_taskloop_clauses + (OmpinReduction,)), **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpTaskConstruct(OmpConstruct):
    """Class representing a Task Construct """
    def __init__(self, **kwargs):
        super().__init__(OMP_Task_Construct, _valid_task_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpSingleConstruct(OmpConstruct):
    """Class representing a single construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Single_Construct, _valid_single_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpCriticalConstruct(OmpConstruct):
    """Class representing a Critical construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Critical_Construct, (OmpCriticalName,), **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpSimdConstruct(OmpConstruct):
    """Class representing a Simd construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, _valid_simd_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpMasterConstruct(OmpConstruct):
    """Class representing the master construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Master_Construct, _valid_simd_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpMaskedConstruct(OmpConstruct):
    """Class representing a Masked construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Masked_Construct, (OmpFilter,), **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpSectionsConstruct(OmpConstruct):
    """Class representing a Sections construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Sections_Construct, _valid_sections_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpSectionConstruct(OmpConstruct):
    """Class representing a Section construct."""
    def __init__(self, **kwargs):
        super().__init__(MP_Section_Construct, None, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpDistributeConstruct(OmpConstruct):
    """Class representing a distribute construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, _valid_Distribute_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpBarrierConstruct(OmpConstruct):
    """Class representing a Barrier construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, None, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpTaskWaitConstruct(OmpConstruct):
    """Class representing a TaskWait construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, None, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpTaskyieldConstruct(OmpConstruct):
    """Class representing a Taskyield construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, None, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpFlushConstruct(OmpConstruct):
    """Class representing a Flush construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, (FlushList,), **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpCancelConstruct(OmpConstruct):
    """Class representing a Cancel construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Cancel_Construct, (OmpCancelType,), **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpTargetConstruct(OmpConstruct):
    """Class representing a Target construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Target_Construct, _valid_target_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpTeamsConstruct(OmpConstruct):
    """Class representing a Teams construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Teams_Construct, _valid_teams_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpAtomicConstruct(OmpConstruct):
    """Class representing an Atomic construct ."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, _valid_atomic_clauses, **kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpEndClause(BasicStmt):
    """Class representing an end construct."""
    def __init__(self, **kwargs):
        self.construct = kwargs.pop('construct')
        self.simd      = kwargs.pop('simd', '')
        self.nowait    = kwargs.pop('nowait', '')

        construct = ' '.join(self.construct)
        txt = 'end {0} {1} {2}'.format(construct, self.simd, self.nowait)

        self._expr = Omp_End_Clause(txt)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpFinal(BasicStmt):
    """Class representing a final clause"""
    def __init__(self, **kwargs):
        self.final = kwargs.pop('final')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        final = self.final
        return 'final({})'.format(final)

class OmpNumThread(BasicStmt):
    """Class representing a num_thread clause."""
    def __init__(self, **kwargs):
        self.thread = kwargs.pop('thread')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        thread = self.thread
        return 'num_threads({})'.format(thread)

class OmpNumTeams(BasicStmt):
    """Class representing a num_teams clause."""
    def __init__(self, **kwargs):
        self.teams = kwargs.pop('teams')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        teams = self.teams
        return 'num_teams({})'.format(teams)

class OmpThreadLimit(BasicStmt):
    """Class representing a thread_limit clause."""
    def __init__(self, **kwargs):
        self.limit = kwargs.pop('limit')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        limit = self.limit
        return 'thread_limit({})'.format(limit)

class OmpNumTasks(BasicStmt):
    """Class representing a num_tasks clause."""
    def __init__(self, **kwargs):
        self.tasks = kwargs.pop('tasks')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        tasks = self.tasks
        return 'num_tasks({})'.format(tasks)

class OmpGrainSize(BasicStmt):
    """Class representing a grainsize clause."""
    def __init__(self, **kwargs):
        self.size = kwargs.pop('tasks')

        super().__init__(**kwargs)

    @property
    def expr(self):
        # TODO check if variable exist in namespace
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        size = self.size
        return 'grainsize({})'.format(size)

class OmpDefault(BasicStmt):
    """Class representing a default clause."""
    def __init__(self, **kwargs):
        self.status = kwargs.pop('status')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'default({})'.format(self.status)

class OmpProcBind(BasicStmt):
    """Class representing a proc_bind clause."""
    def __init__(self, **kwargs):
        self.status = kwargs.pop('status')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'proc_bind({})'.format(self.status)

class OmpPrivate(BasicStmt):
    """Class representing a private clause."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'private({})'.format(args)

class FlushList(BasicStmt):
    """Class representing a list of variables of the flush construct."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return '({})'.format(args)

class OmpCriticalName(BasicStmt):
    """Class representing the name of a critical construct."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        txt = str(self.args)
        return '({})'.format(txt)

class OmpShared(BasicStmt):
    """Class representing a shared clause."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'shared({})'.format(args)

class OmpFirstPrivate(BasicStmt):
    """Class representing a firstprivate clause."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'firstprivate({})'.format(args)

class OmpLastPrivate(BasicStmt):
    """Class representing a lastprivate clause."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'lastprivate({})'.format(args)

class OmpCopyin(BasicStmt):
    """Class representing a copyin clause."""
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        args = ', '.join(str(arg) for arg in self.args)
        return 'copyin({})'.format(args)

class OmpReduction(BasicStmt):
    """Class representing a reduction clause."""
    def __init__(self, **kwargs):
        self.op   = kwargs.pop('op')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        op   = self.op
        args = ', '.join(str(arg) for arg in self.args)
        return 'reduction({0}: {1})'.format(op, args)

class OmpDepend(BasicStmt):
    """Class representing a depend clause."""
    def __init__(self, **kwargs):
        self.dtype   = kwargs.pop('dtype')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        dtype   = self.dtype
        args = ', '.join(str(arg) for arg in self.args)
        return 'depend({0}: {1})'.format(dtype, args)

class OmpMap(BasicStmt):
    """Class representing a map clause."""
    def __init__(self, **kwargs):
        self.mtype   = kwargs.pop('mtype')
        self.args = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        mtype   = self.mtype
        args = ', '.join(str(arg) for arg in self.args)
        return 'map({0} {1})'.format(mtype, args)

class OmpinReduction(BasicStmt):
    """Class representing an in_reduction clause."""
    def __init__(self, **kwargs):
        self.ctype  = kwargs.pop('ctype')
        self.op     = kwargs.pop('op')
        self.args   = kwargs.pop('args')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        # TODO check if variable exist in namespace
        ctype = self.ctype
        op    = self.op
        args  = ', '.join(str(arg) for arg in self.args)
        return '{0}({1}: {2})'.format(ctype, op, args)

class OmpCollapse(BasicStmt):
    """Class representing a collapse clause."""
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'collapse({})'.format(self.n)

class OmpOrdered(BasicStmt):
    """Class representing an ordered clause."""
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        if self.n:
            return 'ordered({})'.format(self.n)
        else:
            return 'ordered'

class OmpLinear(BasicStmt):
    """Class representing a linear clause."""
    def __init__(self, **kwargs):
        self.val  = kwargs.pop('val')
        self.step = kwargs.pop('step')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'linear({0}:{1})'.format(self.val, self.step)

class OmpSchedule(BasicStmt):
    """Class representing a schedule clause."""
    def __init__(self, **kwargs):
        self.kind       = kwargs.pop('kind')
        self.chunk_size = kwargs.pop('chunk_size', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        if self.chunk_size:
            return 'schedule({0}, {1})'.format(self.kind, self.chunk_size)
        else:
            return 'schedule({0})'.format(self.kind)

class OmpFilter(BasicStmt):
    """Class representing a filter clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.n = kwargs.pop('n')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return '{}({})'.format(self.name, self.n)

class OmpUntied(BasicStmt):
    """Class representing an untied clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'untied'

class OmpMergeable(BasicStmt):
    """Class representing a mergeable clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'mergeable'

class OmpNogroup(BasicStmt):
    """Class representing a nogroup clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return 'nogroup'

class OmpPriority(BasicStmt):
    """Class representing a priority clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.n = kwargs.pop('n')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return '{}({})'.format(self.name, self.n)

class OmpAtomicClause(BasicStmt):
    """Class representing an atomic clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return '{}'.format(self.name)

class OmpCancelType(BasicStmt):
    """Class representing a type of a cancel construct."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return '{}'.format(self.name)

class AtomicMemoryClause(BasicStmt):
    """Class representing a atomic memory clause."""
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return '{}'.format(self.name)

class OmpForSimd(BasicStmt):
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
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
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
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
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
        self.sname = kwargs.pop('sname')

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Sections: expr")

        txt = self.sname
        return txt

class OmpTaskloopSimd(BasicStmt):
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
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

class OmpDistributeCombined(BasicStmt):
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
        self.dname  = kwargs.pop('dname')
        self.sname  = kwargs.pop('sname', None)
        self.pname  = kwargs.pop('pname', None)
        self.fname  = kwargs.pop('fname', None)
        self.ssname = kwargs.pop('ssname', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Teams Distribute: expr")

        txt = self.dname
        if self.sname:
            txt += ' ' + self.sname
        elif self.pname:
            txt += ' ' + self.pname
            txt += ' ' + self.fname
            if self.ssname:
                txt += ' ' + self.ssname

        return txt

class OmpTargetParallel(BasicStmt):
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
        self.pname = kwargs.pop('pname')
        self.fname = kwargs.pop('fname', None)
        self.sname = kwargs.pop('sname', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Target Parallel: expr")

        txt = self.pname
        if self.fname:
            txt += ' ' + self.fname
            if self.sname:
                txt += ' ' + self.sname

        return txt

class OmpTargetTeams(BasicStmt):
    """Class representing a combined comstruct."""
    def __init__(self, **kwargs):
        self.tname  = kwargs.pop('tname')
        self.dname  = kwargs.pop('dname', None)
        self.sname  = kwargs.pop('sname', None)
        self.pname  = kwargs.pop('pname', None)
        self.fname  = kwargs.pop('fname', None)
        self.ssname = kwargs.pop('ssname', None)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> Combined Target Teams: expr")

        txt = self.tname
        if self.dname:
            txt += ' ' + self.dname
            if self.sname:
                txt += ' ' + self.sname
            else:
                txt += ' ' + self.pname + ' ' + self.fname
                if self.ssname:
                    txt += ' ' + self.ssname
        return txt

#################################################

#################################################
# whenever a new rule (construct/clause) is added in the grammar, we must update the following tuples.

_valid_single_clauses = (OmpPrivate,
                         OmpFirstPrivate)

_valid_atomic_clauses = (OmpAtomicClause,
                         AtomicMemoryClause)

_valid_task_clauses = (OmpPriority,
                       OmpFinal,
                       OmpDefault,
                       OmpPrivate,
                       OmpShared,
                       OmpFirstPrivate,
                       OmpUntied,
                       OmpMergeable,
                       OmpinReduction,
                       OmpDepend)

_valid_target_clauses = (OmpPrivate,
                         OmpLastPrivate,
                         OmpinReduction,
                         OmpDepend,
                         OmpMap)

_valid_teams_clauses = (OmpPrivate,
                        OmpLastPrivate,
                        OmpShared,
                        OmpReduction,
                        OmpNumTeams,
                        OmpThreadLimit)

_valid_sections_clauses = (OmpPrivate,
                           OmpFirstPrivate,
                           OmpLastPrivate,
                           OmpReduction)

_valid_Distribute_clauses = (OmpPrivate,
                             OmpFirstPrivate,
                             OmpLastPrivate,
                             OmpCollapse)

_valid_simd_clauses = (OmpLinear,
                       OmpReduction,
                       OmpCollapse,
                       OmpLastPrivate)

_valid_taskloop_clauses = (OmpShared,
                           OmpPrivate,
                           OmpFirstPrivate,
                           OmpLastPrivate,
                           OmpReduction,
                           OmpinReduction,
                           OmpNumTasks,
                           OmpGrainSize,
                           OmpCollapse,
                           OmpUntied,
                           OmpMergeable,
                           OmpNogroup,
                           OmpPriority)

_valid_loop_clauses = (OmpPrivate,
                       OmpFirstPrivate,
                       OmpLastPrivate,
                       OmpReduction,
                       OmpSchedule,
                       OmpCollapse,
                       OmpLinear,
                       OmpOrdered)

_valid_parallel_clauses = (OmpNumThread,
                           OmpDefault,
                           OmpPrivate,
                           OmpShared,
                           OmpFirstPrivate,
                           OmpCopyin,
                           OmpReduction,
                           OmpProcBind)

omp_directives = (OmpParallelConstruct,
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
                  OmpSectionConstruct)

omp_clauses = (OmpCollapse,
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
               OmpFinal,
               OmpCancelType,
               OmpMap,
               OmpNumTeams,
               OmpThreadLimit,
               OmpForSimd,
               OmpMaskedTaskloop,
               OmpPSections,
               OmpTaskloopSimd,
               OmpDistributeCombined,
               OmpTargetParallel,
               OmpTargetTeams)

omp_classes = (Openmp, OpenmpStmt) + omp_directives + omp_clauses

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
