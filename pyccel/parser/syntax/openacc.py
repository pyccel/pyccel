# coding: utf-8

"""
"""

from os.path import join, dirname

from pyccel.parser.syntax.basic import BasicStmt
from pyccel.ast.core import AnnotatedComment

DEBUG = False

class Openacc(object):
    """Class for Openacc syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Openacc.

        """
        self.statements = kwargs.pop('statements', [])

class AccBasic(BasicStmt):
    pass


class OpenaccStmt(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.stmt = kwargs.pop('stmt')

        super(OpenaccStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> OpenaccStmt: expr")

        stmt = self.stmt
        _valid_const = (AccParallelConstruct,
                        AccKernelsConstruct,
                        AccDataConstruct,
                        AccEnterDataDirective,
                        AccExitDataDirective,
                        AccHostDataDirective,
                        AccLoopConstruct,
                        AccAtomicConstruct,
                        AccDeclareDirective,
                        AccInitDirective,
                        AccShutDownDirective,
                        AccSetDirective,
                        AccUpdateDirective,
                        AccRoutineDirective,
                        AccWaitDirective,
                        AccEndClause)

        if isinstance(stmt, _valid_const):
            return stmt.expr
        else:
            raise TypeError('Unexpected construct or directive of type {0}'.format(type(stmt)))

#################################################
#           Constructs and Directives
#################################################
class AccParallelConstruct(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccParallelConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccParallelConstruct: expr")

        _valid_clauses = (AccAsync,
                          AccWait,
                          AccNumGangs,
                          AccNumWorkers,
                          AccVectorLength,
                          AccDeviceType,
                          AccIf,
                          AccReduction,
                          AccCopy,
                          AccCopyin,
                          AccCopyout,
                          AccCreate,
                          AccPresent,
                          AccDevicePtr,
                          AccPrivate,
                          AccFirstPrivate,
                          AccDefault)

        txt = 'parallel'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccKernelsConstruct(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccKernelsConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccKernelsConstruct: expr")

        _valid_clauses = (AccAsync,
                          AccWait,
                          AccNumGangs,
                          AccNumWorkers,
                          AccVectorLength,
                          AccDeviceType,
                          AccIf,
                          AccCopy,
                          AccCopyin,
                          AccCopyout,
                          AccCreate,
                          AccPresent,
                          AccDevicePtr,
                          AccDefault)

        txt = 'kernels'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccDataConstruct(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccDataConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDataConstruct: expr")

        _valid_clauses = (AccIf,
                          AccCopy,
                          AccCopyin,
                          AccCopyout,
                          AccCreate,
                          AccPresent,
                          AccDevicePtr)

        txt = 'data'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccEnterDataDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccEnterDataDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccEnterDataDirective: expr")

        _valid_clauses = (AccIf,
                          AccAsync,
                          AccWait,
                          AccCopyin,
                          AccCreate)

        txt = 'enter data'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccExitDataDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccExitDataDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccExitDataDirective: expr")

        _valid_clauses = (AccIf,
                          AccAsync,
                          AccWait,
                          AccCopyout,
                          AccDelete,
                          AccFinalize)

        txt = 'exit data'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccHostDataDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccHostDataDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccHostDataDirective: expr")

        _valid_clauses = (AccUseDevice)

        txt = 'host_data'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccLoopConstruct(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccLoopConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccLoopConstruct: expr")

        _valid_clauses = (AccCollapse,
                          AccGang,
                          AccWorker,
                          AccVector,
                          AccSeq,
                          AccAuto,
                          AccTile,
                          AccDeviceType,
                          AccIndependent,
                          AccPrivate,
                          AccReduction)

        txt = 'loop'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccAtomicConstruct(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccAtomicConstruct, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccAtomicConstruct: expr")

        txt = 'atomic'
        for clause in self.clauses:
            if isinstance(clause, AccAtomicClause):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccDeclareDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccDeclareDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDeclareDirective: expr")

        _valid_clauses = (AccCopy,
                          AccCopyin,
                          AccCopyout,
                          AccCreate,
                          AccPresent,
                          AccDevicePtr,
                          AccDeviceResident,
                          AccLink)

        txt = 'declare'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccInitDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccInitDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccInitDirective: expr")

        _valid_clauses = (AccDeviceType,
                          AccDeviceNum)

        txt = 'init'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccShutDownDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccShutDownDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccShutDownDirective: expr")

        _valid_clauses = (AccDeviceType,
                          AccDeviceNum)

        txt = 'shutdown'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccSetDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccSetDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccSetDirective: expr")

        _valid_clauses = (AccDefaultAsync,
                          AccDeviceType,
                          AccDeviceNum)

        txt = 'set'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccUpdateDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccUpdateDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccUpdateDirective: expr")

        _valid_clauses = (AccAsync,
                          AccWait,
                          AccDeviceType,
                          AccIf,
                          AccIfPresent,
                          AccSelf,
                          AccHost,
                          AccDevice)

        txt = 'update'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccRoutineDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccRoutineDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccRoutineDirective: expr")

        _valid_clauses = (AccGang,
                          AccWorker,
                          AccVector,
                          AccSeq,
                          AccBind,
                          AccDeviceType,
                          AccNoHost)

        txt = 'routine'
        for clause in self.clauses:
            if isinstance(clause, _valid_clauses):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccWaitDirective(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clauses = kwargs.pop('clauses')

        super(AccWaitDirective, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccWaitDirective: expr")

        txt = 'wait'
        for clause in self.clauses:
            if isinstance(clause, AccAsync):
                txt = '{0} {1}'.format(txt, clause.expr)
            else:
                raise TypeError('Unexpected clause of type {0}'.format(type(clause)))

        return AnnotatedComment('acc', txt)

class AccEndClause(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.construct = kwargs.pop('construct')

        super(AccEndClause, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccEndClause: expr")

        txt = 'end {0}'.format(self.construct)
        return AnnotatedComment('acc', txt)
#################################################

#################################################
#                 Clauses
#################################################
#AccAsync: 'async' '(' args+=ID[','] ')';
class AccAsync(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccAsync, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccAsync: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'async({})'.format(args)

#AccAuto: 'auto';
class AccAuto(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(AccAuto, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccAuto: expr")

        return 'auto'

#AccBind: 'bind' '(' arg=STRING ')';
class AccBind(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.arg = kwargs.pop('arg')

        super(AccBind, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccBind: expr")

        # TODO check if variable exist in namespace
        arg = self.arg
        return 'bind({})'.format(str(arg))

#AccCache: 'cache' '(' args+=ID[','] ')';
class AccCache(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccCache, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccCache: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'cache({})'.format(args)

#AccCollapse: 'collapse' '(' n=INT ')';
class AccCollapse(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(AccCollapse, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccCollapse: expr")

        return 'collapse({})'.format(self.n)

#AccCopy: 'copy' '(' args+=ID[','] ')';
class AccCopy(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccCopy, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccCopy: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'copy({})'.format(args)

#AccCopyin: 'copyin' '(' args+=ID[','] ')';
class AccCopyin(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccCopyin, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccCopyin: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'copyin({})'.format(args)

#AccCopyout: 'copyout' '(' args+=ID[','] ')';
class AccCopyout(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccCopyout, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccCopyout: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'copyout({})'.format(args)

#AccCreate: 'create' '(' args+=ID[','] ')';
class AccCreate(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccCreate, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccCreate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'create({})'.format(args)

#AccDefault: 'default' '(' status=DefaultStatus ')';
class AccDefault(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.status = kwargs.pop('status')

        super(AccDefault, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDefault: expr")

        return 'default({})'.format(self.status)

#AccDefaultAsync: 'default_async' '(' args+=ID[','] ')';
class AccDefaultAsync(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccDefaultAsync, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDefaultAsync: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'default_async({})'.format(args)

#AccDelete: 'delete' '(' args+=ID[','] ')';
class AccDelete(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccDelete, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDelete: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'delete({})'.format(args)

#AccDevice: 'device' '(' args+=ID[','] ')';
class AccDevice(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccDevice, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDevice: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'device({})'.format(args)

#AccDeviceNum: 'device_num' '(' n=INT ')';
class AccDeviceNum(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(AccDeviceNum, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDeviceNum: expr")

        return 'device_num({})'.format(self.n)

#AccDevicePtr: 'deviceptr' '(' args+=ID[','] ')';
class AccDevicePtr(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccDevicePtr, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDevicePtr: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'deviceptr({})'.format(args)

#AccDeviceResident: 'device_resident' '(' args+=ID[','] ')';
class AccDeviceResident(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccDeviceResident, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDeviceResident: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'device_resident({})'.format(args)

#AccDeviceType: 'device_type' '(' args+=ID[','] ')';
class AccDeviceType(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccDeviceType, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccDeviceType: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'device_type({})'.format(args)

#AccFinalize: 'finalize';
class AccFinalize(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(AccFinalize, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccFinalize: expr")

        return 'finalize'

#AccFirstPrivate: 'firstprivate' '(' args+=ID[','] ')';
class AccFirstPrivate(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccFirstPrivate, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccFirstPrivate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'firstprivate({})'.format(args)

#AccGang: 'gang' '(' args+=GangArg[','] ')';
class AccGang(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccGang, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccGang: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(a.arg) for a in self.args)
        return 'gang({})'.format(args)

#AccHost: 'host' '(' args+=ID[','] ')';
class AccHost(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccHost, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccHost: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'host({})'.format(args)

#AccIf: 'if' cond=ID;
class AccIf(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.cond = kwconds.pop('cond')

        super(AccIf, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccIf: expr")

        # TODO check if variable exist in namespace
        cond = self.cond
        return 'if({})'.format(str(cond))

#AccIfPresent: 'if_present';
class AccIfPresent(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(AccIfPresent, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccIfPresent: expr")

        return 'if_present'

#AccIndependent: 'independent';
class AccIndependent(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(AccIndependent, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccIndependent: expr")

        return 'independent'

#AccLink: 'link' '(' args+=ID[','] ')';
class AccLink(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccLink, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccLink: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'link({})'.format(args)

#AccNoHost: 'nohost';
class AccNoHost(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(AccNoHost, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccNoHost: expr")

        return 'nohost'

#AccNumGangs: 'num_gangs' '(' n=INT ')';
class AccNumGangs(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(AccNumGangs, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccNumGangs: expr")

        return 'num_gangs({})'.format(self.n)

#AccNumWorkers: 'num_workers' '(' n=INT ')';
class AccNumWorkers(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(AccNumWorkers, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccNumWorkers: expr")

        return 'num_workers({})'.format(self.n)

#AccPresent: 'present' '(' args+=ID[','] ')';
class AccPresent(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccPresent, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccPresent: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'present({})'.format(args)

#AccPrivate: 'private' '(' args+=ID[','] ')';
class AccPrivate(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccPrivate, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccPrivate: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'private({})'.format(args)

#AccReduction: 'reduction' '('op=ReductionOperator ':' args+=ID[','] ')';
class AccReduction(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.op   = kwargs.pop('op')
        self.args = kwargs.pop('args')

        super(AccReduction, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccReduction: expr")

        # TODO check if variable exist in namespace
        op   = self.op
        args = ', '.join(str(arg) for arg in self.args)
        return 'reduction({0}: {1})'.format(op, args)

#AccSelf: 'self' '(' args+=ID[','] ')';
class AccSelf(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccSelf, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccSelf: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'self({})'.format(args)

#AccSeq: 'seq';
class AccSeq(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(AccSeq, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccSeq: expr")

        return 'seq'

#AccTile: 'tile' '(' args+=ID[','] ')';
class AccTile(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccTile, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccTile: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'tile({})'.format(args)

#AccUseDevice: 'use_device' '(' args+=ID[','] ')';
class AccUseDevice(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccUseDevice, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccUseDevice: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'use_device({})'.format(args)

#AccVector: 'vector' ('(' args+=VectorArg ')')?;
class AccVector(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccVector, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccVector: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(a.arg) for a in self.args)
        return 'vector({})'.format(args)

#AccVectorLength: 'vector_length' '(' n=INT ')';
class AccVectorLength(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.n = kwargs.pop('n')

        super(AccVectorLength, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccVectorLength: expr")

        return 'vector_length({})'.format(self.n)

#AccWait: 'wait' '(' args+=ID[','] ')';
class AccWait(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccWait, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccWait: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(arg) for arg in self.args)
        return 'wait({})'.format(args)

#AccWorker: 'worker' ('(' args+=WorkerArg ')')?;
class AccWorker(AccBasic):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args')

        super(AccWorker, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print("> AccWorker: expr")

        # TODO check if variable exist in namespace
        args = ', '.join(str(a.arg) for a in self.args)
        return 'worker({})'.format(args)
#################################################

#################################################
# whenever a new rule is added in the grammar, we must update the following
# lists.
acc_directives = [AccParallelConstruct,
                  AccKernelsConstruct,
                  AccDataConstruct,
                  AccEnterDataDirective,
                  AccExitDataDirective,
                  AccHostDataDirective,
                  AccLoopConstruct,
                  AccAtomicConstruct,
                  AccDeclareDirective,
                  AccInitDirective,
                  AccShutDownDirective,
                  AccSetDirective,
                  AccUpdateDirective,
                  AccRoutineDirective,
                  AccWaitDirective,
                  AccEndClause]

acc_clauses = [AccAsync,
               AccAuto,
               AccBind,
               AccCollapse,
               AccCopy,
               AccCopyin,
               AccCopyout,
               AccCreate,
               AccDefault,
               AccDefaultAsync,
               AccDelete,
               AccDevice,
               AccDeviceNum,
               AccDevicePtr,
               AccDeviceResident,
               AccDeviceType,
               AccFinalize,
               AccFirstPrivate,
               AccGang,
               AccHost,
               AccIf,
               AccIfPresent,
               AccIndependent,
               AccLink,
               AccNoHost,
               AccNumGangs,
               AccNumWorkers,
               AccPresent,
               AccPrivate,
               AccReduction,
               AccSelf,
               AccSeq,
               AccTile,
               AccUseDevice,
               AccVector,
               AccVectorLength,
               AccWait,
               AccWorker]

acc_classes = [Openacc, OpenaccStmt] + acc_directives + acc_clauses

def parse(filename=None, stmts=None, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, '../grammar/openacc.tx')

    from textx.metamodel import metamodel_from_file
    meta = metamodel_from_file(grammar, debug=debug, classes=acc_classes)

    # Instantiate model
    if filename:
        model = meta.model_from_file(filename)
    elif stmts:
        model = meta.model_from_str(stmts)
    else:
        raise ValueError('Expecting a filename or a string')

    stmts = []
    for stmt in model.statements:
        if isinstance(stmt, OpenaccStmt):
            e = stmt.stmt.expr
            stmts.append(e)

    if len(stmts) == 1:
        return stmts[0]
    else:
        return stmts
#################################################
