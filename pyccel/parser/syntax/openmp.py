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
                            OMP_Sections_Construct, OMP_Section_Construct, OMP_Simd_Construct,
                            OMP_Distribute_Construct, OMP_TaskLoop_Construct)

DEBUG = False

class OmpConstruct(BasicStmt):
    """Class representing all OpenMP constructs."""
    def __init__(self, omp_type, vclauses, **kwargs):
        name     = kwargs.pop('name', None)
        clauses  = kwargs.pop('clauses', None)
        combined = kwargs.pop('combined', None)
        simd     = kwargs.pop('combinedsimd', None)

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

        has_nowait = False
        txt = ''
        if name:
            txt += name
        if simd:
            _valid_clauses += _valid_simd_clauses
            txt += ' ' + simd
        if clauses:
            clause_expr, has_nowait = check_get_clauses(self, _valid_clauses, clauses, combined)
            txt += clause_expr

        if combined:
            self._expr = omp_type(txt, has_nowait, com)
        else:
            self._expr = omp_type(txt, has_nowait)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

class OmpClauses(BasicStmt):
    """Class representing the clause expr."""
    _expr = None
    @property
    def expr(self):
        if DEBUG:
            print("> {}: expr".format(type(self).__name__))

        return self._expr

def check_get_clauses(name, valid_clauses, clauses, combined = None):
    """
    Function to check if the clauses are correct for a given construct.
    And set the has_nowait variable to True if there is a nowait clause, to finally add the nowait clause at the end of the construct.
    """
    has_nowait = False
    txt = ''
    for clause in clauses:
        if isinstance(clause, valid_clauses) and \
           not (isinstance(clause, OmpCopyin) and isinstance(combined, OmpTargetParallel)):
            if isinstance(clause, OmpNowait):
                if isinstance(name, (OmpLoopConstruct, OmpSectionsConstruct, OmpSingleConstruct)):
                    has_nowait = True
                else:
                    raise TypeError("Wrong clause nowait")
            else:
                txt = '{0} {1}'.format(txt, clause.expr)
        else:
            msg = "Wrong clause " + type(clause).__name__
            raise TypeError(msg)
    return txt, has_nowait


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
    """Class representing the Parallel construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Parallel_Construct, _valid_parallel_clauses, **kwargs)

class OmpLoopConstruct(OmpConstruct):
    """Class representing the For loop construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_For_Loop, _valid_loop_clauses, **kwargs)

class OmpTaskLoopConstruct(OmpConstruct):
    """Class representing the Taskloop construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_TaskLoop_Construct, (_valid_taskloop_clauses + (OmpinReduction,)), **kwargs)

class OmpTaskConstruct(OmpConstruct):
    """Class representing the Task construct """
    def __init__(self, **kwargs):
        super().__init__(OMP_Task_Construct, _valid_task_clauses, **kwargs)

class OmpSingleConstruct(OmpConstruct):
    """Class representing the Single construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Single_Construct, _valid_single_clauses, **kwargs)

class OmpCriticalConstruct(OmpConstruct):
    """Class representing the Critical construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Critical_Construct, (OmpCriticalName,), **kwargs)

class OmpSimdConstruct(OmpConstruct):
    """Class representing the Simd construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Simd_Construct, _valid_simd_clauses, **kwargs)

class OmpMasterConstruct(OmpConstruct):
    """Class representing the master construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Master_Construct, _valid_simd_clauses, **kwargs)

class OmpMaskedConstruct(OmpConstruct):
    """Class representing the Masked construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Masked_Construct, (OmpFilter,), **kwargs)

class OmpSectionsConstruct(OmpConstruct):
    """Class representing the Sections construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Sections_Construct, _valid_sections_clauses, **kwargs)

class OmpSectionConstruct(OmpConstruct):
    """Class representing the Section construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Section_Construct, None, **kwargs)

class OmpDistributeConstruct(OmpConstruct):
    """Class representing the Distribute construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Distribute_Construct, _valid_Distribute_clauses, **kwargs)

class OmpBarrierConstruct(OmpConstruct):
    """Class representing the Barrier construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, None, **kwargs)

class OmpTaskWaitConstruct(OmpConstruct):
    """Class representing the TaskWait construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, None, **kwargs)

class OmpTaskyieldConstruct(OmpConstruct):
    """Class representing the Taskyield construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, None, **kwargs)

class OmpFlushConstruct(OmpConstruct):
    """Class representing the Flush construct."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, (FlushList,), **kwargs)

class OmpCancelConstruct(OmpConstruct):
    """Class representing the Cancel construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Cancel_Construct, (OmpCancelType,), **kwargs)

class OmpTargetConstruct(OmpConstruct):
    """Class representing the Target construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Target_Construct, _valid_target_clauses, **kwargs)

class OmpTeamsConstruct(OmpConstruct):
    """Class representing the Teams construct."""
    def __init__(self, **kwargs):
        super().__init__(OMP_Teams_Construct, _valid_teams_clauses, **kwargs)

class OmpAtomicConstruct(OmpConstruct):
    """Class representing the Atomic construct ."""
    def __init__(self, **kwargs):
        super().__init__(OmpAnnotatedComment, _valid_atomic_clauses, **kwargs)

class OmpEndClause(BasicStmt):
    """Class representing the End construct."""
    def __init__(self, **kwargs):
        lst_construct = kwargs.pop('construct')

        construct = ' '.join(lst_construct)
        txt = 'end {0}'.format(construct)

        self._expr = Omp_End_Clause(txt, False)

        super().__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OmpEndClause: expr")

        return self._expr

class OmpFinal(OmpClauses):
    """Class representing the final clause"""
    def __init__(self, **kwargs):
        final = kwargs.pop('final')

        self._expr = 'final({})'.format(final)

        super().__init__(**kwargs)

class OmpNumThread(OmpClauses):
    """Class representing the num_thread clause."""
    def __init__(self, **kwargs):
        thread = kwargs.pop('thread')

        self._expr = 'num_threads({})'.format(thread)

        super().__init__(**kwargs)

class OmpNumTeams(OmpClauses):
    """Class representing the num_teams clause."""
    def __init__(self, **kwargs):
        teams = kwargs.pop('teams')

        self._expr = 'num_teams({})'.format(teams)

        super().__init__(**kwargs)

class OmpThreadLimit(OmpClauses):
    """Class representing the thread_limit clause."""
    def __init__(self, **kwargs):
        limit = kwargs.pop('limit')

        self._expr = 'thread_limit({})'.format(limit)

        super().__init__(**kwargs)

class OmpNumTasks(OmpClauses):
    """Class representing the num_tasks clause."""
    def __init__(self, **kwargs):
        tasks = kwargs.pop('tasks')

        self._expr = 'num_tasks({})'.format(tasks)

        super().__init__(**kwargs)

class OmpGrainSize(OmpClauses):
    """Class representing the grainsize clause."""
    def __init__(self, **kwargs):
        size = kwargs.pop('tasks')

        self._expr = 'grainsize({})'.format(size)

        super().__init__(**kwargs)

class OmpDefault(OmpClauses):
    """Class representing the default clause."""
    def __init__(self, **kwargs):
        status = kwargs.pop('status')

        self._expr = 'default({})'.format(status)

        super().__init__(**kwargs)

class OmpProcBind(OmpClauses):
    """Class representing the proc_bind clause."""
    def __init__(self, **kwargs):
        status = kwargs.pop('status')

        self._expr = 'proc_bind({})'.format(status)

        super().__init__(**kwargs)

class OmpPrivate(OmpClauses):
    """Class representing the private clause."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        #TODO check for variables in namespace
        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'private({})'.format(txt)

        super().__init__(**kwargs)

class FlushList(OmpClauses):
    """Class representing a list of variables for the flush construct."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = '({})'.format(txt)

        super().__init__(**kwargs)

class OmpCriticalName(OmpClauses):
    """Class representing the name of a critical construct."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        self._expr = '({})'.format(str(args))

        super().__init__(**kwargs)

class OmpShared(OmpClauses):
    """Class representing the shared clause."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'shared({})'.format(txt)

        super().__init__(**kwargs)

class OmpFirstPrivate(OmpClauses):
    """Class representing the firstprivate clause."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'firstprivate({})'.format(txt)

        super().__init__(**kwargs)

class OmpLastPrivate(OmpClauses):
    """Class representing the lastprivate clause."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'lastprivate({})'.format(txt)

        super().__init__(**kwargs)

class OmpCopyin(OmpClauses):
    """Class representing the copyin clause."""
    def __init__(self, **kwargs):
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'copyin({})'.format(txt)

        super().__init__(**kwargs)

class OmpReduction(OmpClauses):
    """Class representing the reduction clause."""
    def __init__(self, **kwargs):
        op   = kwargs.pop('op')
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'reduction({0}: {1})'.format(op, txt)

        super().__init__(**kwargs)

class OmpDepend(OmpClauses):
    """Class representing the depend clause."""
    def __init__(self, **kwargs):
        dtype   = kwargs.pop('dtype')
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'depend({0}: {1})'.format(dtype, txt)

        super().__init__(**kwargs)

class OmpMap(OmpClauses):
    """Class representing the map clause."""
    def __init__(self, **kwargs):
        mtype   = kwargs.pop('mtype')
        args = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = 'map({0} {1})'.format(mtype, txt)

        super().__init__(**kwargs)

class OmpinReduction(OmpClauses):
    """Class representing the in_reduction clause."""
    def __init__(self, **kwargs):
        ctype  = kwargs.pop('ctype')
        op     = kwargs.pop('op')
        args   = kwargs.pop('args')

        txt = ', '.join(str(arg) for arg in args)
        self._expr = '{0}({1}: {2})'.format(ctype, op, txt)

        super().__init__(**kwargs)

class OmpCollapse(OmpClauses):
    """Class representing the collapse clause."""
    def __init__(self, **kwargs):
        n = kwargs.pop('n')

        self._expr = 'collapse({})'.format(n)

        super().__init__(**kwargs)

class OmpOrdered(OmpClauses):
    """Class representing the ordered clause."""
    def __init__(self, **kwargs):
        n = kwargs.pop('n', None)

        if n:
            self._expr = 'ordered({})'.format(n)
        else:
            self._expr = 'ordered'

        super().__init__(**kwargs)

class OmpLinear(OmpClauses):
    """Class representing the linear clause."""
    def __init__(self, **kwargs):
        val  = kwargs.pop('val')
        step = kwargs.pop('step')

        self._expr = 'linear({0}:{1})'.format(val, step)

        super().__init__(**kwargs)

class OmpSchedule(OmpClauses):
    """Class representing the schedule clause."""
    def __init__(self, **kwargs):
        kind       = kwargs.pop('kind')
        chunk_size = kwargs.pop('chunk_size', None)

        if chunk_size:
            self._expr = 'schedule({0}, {1})'.format(kind, chunk_size)
        else:
            self._expr = 'schedule({0})'.format(kind)

        super().__init__(**kwargs)

class OmpFilter(OmpClauses):
    """Class representing the filter clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')
        n = kwargs.pop('n')

        self._expr = '{}({})'.format(name, n)

        super().__init__(**kwargs)

class OmpUntied(OmpClauses):
    """Class representing the untied clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class OmpMergeable(OmpClauses):
    """Class representing the mergeable clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class OmpNogroup(OmpClauses):
    """Class representing the nogroup clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class OmpPriority(OmpClauses):
    """Class representing the priority clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')
        n = kwargs.pop('n')

        self._expr = '{}({})'.format(name, n)

        super().__init__(**kwargs)

class OmpAtomicClause(OmpClauses):
    """Class representing the atomic clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class OmpCancelType(OmpClauses):
    """Class representing the type of the cancel construct."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class AtomicMemoryClause(OmpClauses):
    """Class representing the atomic memory clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class OmpNowait(OmpClauses):
    """Class representing the nowait clause."""
    def __init__(self, **kwargs):
        name = kwargs.pop('name')

        self._expr = name

        super().__init__(**kwargs)

class OmpForSimd(OmpClauses):
    """Class representing the combined For Simd construct."""
    def __init__(self, **kwargs):
        fname = kwargs.pop('fname')
        sname = kwargs.pop('sname', None)

        txt = fname
        if sname:
            txt = txt + ' ' + sname
        self._expr = '{}'.format(txt)

        super().__init__(**kwargs)

class OmpMaskedTaskloop(OmpClauses):
    """Class representing the combined Masked Taskloop construct."""
    def __init__(self, **kwargs):
        mname = kwargs.pop('mname')
        tname = kwargs.pop('tname', None)
        sname = kwargs.pop('sname', None)

        txt = mname
        if tname:
            txt = txt + ' ' + tname
            if sname:
                txt = txt + ' ' + sname
        self._expr = '{}'.format(txt)

        super().__init__(**kwargs)

class OmpPSections(OmpClauses):
    """Class representing the combined Parallel Sections construct."""
    def __init__(self, **kwargs):
        sname = kwargs.pop('sname')

        self._expr = sname

        super().__init__(**kwargs)

class OmpTaskloopSimd(OmpClauses):
    """Class representing the combined Taskloop Simd comstruct."""
    def __init__(self, **kwargs):
        tname = kwargs.pop('tname')
        sname = kwargs.pop('sname', None)

        txt = tname
        if sname:
            txt += ' ' + sname
        self._expr = txt

        super().__init__(**kwargs)

class OmpDistributeCombined(OmpClauses):
    """Class representing the combined Distribute construct."""
    def __init__(self, **kwargs):
        dname  = kwargs.pop('dname')
        sname  = kwargs.pop('sname', None)
        pname  = kwargs.pop('pname', None)
        fname  = kwargs.pop('fname', None)
        ssname = kwargs.pop('ssname', None)

        txt = dname
        if sname:
            txt += ' ' + sname
        elif pname:
            txt += ' ' + pname
            txt += ' ' + fname
            if ssname:
                txt += ' ' + ssname
        self._expr = txt

        super().__init__(**kwargs)

class OmpTargetParallel(OmpClauses):
    """Class representing the combined Target Parallel construct."""
    def __init__(self, **kwargs):
        pname = kwargs.pop('pname')
        fname = kwargs.pop('fname', None)
        sname = kwargs.pop('sname', None)

        txt = pname
        if fname:
            txt = ' ' + fname
            if sname:
                txt += ' ' + sname
        self._expr = txt

        super().__init__(**kwargs)

class OmpTargetTeams(OmpClauses):
    """Class representing the combined Target Teams construct."""
    def __init__(self, **kwargs):
        tname  = kwargs.pop('tname')
        dname  = kwargs.pop('dname', None)
        sname  = kwargs.pop('sname', None)
        pname  = kwargs.pop('pname', None)
        fname  = kwargs.pop('fname', None)
        ssname = kwargs.pop('ssname', None)

        txt = tname
        if dname:
            txt += ' ' + dname
            if sname:
                txt += ' ' + sname
            else:
                txt += ' ' + pname + ' ' + fname
                if ssname:
                    txt += ' ' + ssname
        self._expr = txt

        super().__init__(**kwargs)

#################################################

#################################################
# whenever a new rule (construct/clause) is added in the grammar, we must update the following tuples.

_valid_single_clauses = (OmpPrivate,
                         OmpFirstPrivate,
                         OmpNowait)

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
                           OmpReduction,
                           OmpNowait)

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
                       OmpOrdered,
                       OmpNowait)

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
               OmpTargetTeams,
               OmpNowait)

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
