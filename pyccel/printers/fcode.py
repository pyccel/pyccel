# coding: utf-8

"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""

from __future__ import print_function, division
import string
from itertools import groupby

from sympy.core import Symbol
from sympy.core import Float
from sympy.core import S, Add, N
from sympy.core import Tuple
from sympy.core.function import Function
from sympy.core.compatibility import string_types
from sympy.printing.precedence import precedence
from sympy.sets.fancysets import Range

from pyccel.types.ast import (Assign, MultiAssign, \
                              Variable, Declare, \
                              Len, Dot, Sign, subs, \
                              IndexedElement, Slice)
from pyccel.printers.codeprinter import CodePrinter

from pyccel.parallel.mpi import MPI
from pyccel.parallel.mpi import MPI_ERROR, MPI_STATUS
from pyccel.parallel.mpi import MPI_comm_world, MPI_status_size, MPI_proc_null
from pyccel.parallel.mpi import MPI_comm_size, MPI_comm_rank
from pyccel.parallel.mpi import MPI_comm_recv, MPI_comm_send
from pyccel.parallel.mpi import MPI_comm_irecv, MPI_comm_isend
from pyccel.parallel.mpi import MPI_comm_sendrecv
from pyccel.parallel.mpi import MPI_waitall


# TODO: add examples
# TODO: use _get_statement when returning a string

__all__ = ["FCodePrinter", "fcode"]

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "abs",
    "sign": "sign",
    "conjugate": "conjg"
}

_default_methods = {
    '__init__': 'create',
    '__del__' : 'free',
}

class FCodePrinter(CodePrinter):
    """A printer to convert sympy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'human': True,
        'source_format': 'fixed',
        'tabwidth': 2,
        'contract': True,
        'standard': 77
    }

    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    _relationals = {
        '!=': '/=',
    }

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {0}".format(text)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    # ============ Elements ============ #

    def _print_Module(self, expr):
        return '\n\n'.join(self._print(i) for i in expr.body)

    def _print_Import(self, expr):
        fil = self._print(expr.fil)
        if not expr.funcs:
            return 'use {0}'.format(fil)
        else:
            funcs = ', '.join(self._print(f) for f in expr.funcs)
            return 'use {0}, only: {1}'.format(fil, funcs)

    def _print_Print(self, expr):
        Str=[]
        for f in expr.expr:
             if isinstance(f,str):
                 Str.append(repr(f))
             else:
                Str.append(self._print(f))


        fs = ','.join(Str)

        return 'print * ,{0} '.format(fs)

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '! {0} '.format(txt)

    def _print_AnnotatedComment(self, expr):
        accel = self._print(expr.accel)
        txt   = str(expr.txt)
        return '!${0} {1}'.format(accel, txt)

    def _print_ThreadID(self, expr):
        lhs_code = self._print(expr.lhs)
        func = 'omp_get_thread_num'
        code = "{0} = {1}()".format(lhs_code, func)
        return self._get_statement(code)

    def _print_ThreadsNumber(self, expr):
        lhs_code = self._print(expr.lhs)
        func = 'omp_get_num_threads'
        code = "{0} = {1}()".format(lhs_code, func)
        return self._get_statement(code)


    def _print_Tuple(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '(/ {0} /)'.format(fs)

    def _print_Variable(self, expr):
        name = expr.name
        if isinstance(name, str):
            return name
        else:
            return '%'.join(self._print(n) for n in name)

    def _print_DottedVariable(self, expr):
        name = expr.name
        return '%'.join(self._print(n) for n in name)

    def _print_Stencil(self, expr):
        lhs_code = self._print(expr.lhs)

        if isinstance(expr.shape, Tuple):
            # this is a correction. problem on LRZ
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
        elif isinstance(expr.shape,str):
            shape_code = '0:' + self._print(expr.shape) + '-1'
        else:
            raise TypeError('Unknown type of shape'+str(type(expr.shape)))

        if isinstance(expr.step, Tuple):
            # this is a correction. problem on LRZ
            step_code = ', '.join('-' + self._print(i) + ':' + self._print(i) \
                                  for i in expr.step)
        elif isinstance(expr.step,str):
            step_code = '-' + self._print(expr.step) + ':' + self._print(expr.step)
        else:
            raise TypeError('Unknown type of step'+str(type(expr.step)))

        code ="allocate({0}({1}, {2})) ; {3} = 0".format(lhs_code, shape_code, \
                                                         step_code, lhs_code)
        return self._get_statement(code)

    def _print_Zeros(self, expr):
        lhs_code   = self._print(expr.lhs)

        if isinstance(expr.shape, Tuple):
            # this is a correction. problem on LRZ
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
        elif isinstance(expr.shape,str):
            shape_code = '0:' + self._print(expr.shape) + '-1'
        else:
            raise TypeError('Unknown type of shape'+str(type(expr.shape)))
#        return self._get_statement("%s = zeros(%s)" % (lhs_code, shape_code))
        return self._get_statement("allocate(%s(%s)) ; %s = 0" % (lhs_code, shape_code, lhs_code))

    def _print_Ones(self, expr):
        lhs_code   = self._print(expr.lhs)

        if isinstance(expr.shape, Tuple):
#            shape_code = ', '.join(self._print(i) for i in expr.shape)
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
        else:
            shape_code = '0:' + self._print(expr.shape) + '-1'

#        return self._get_statement("%s = zeros(%s)" % (lhs_code, shape_code))
        return self._get_statement("allocate(%s(%s)) ; %s = 1" % (lhs_code, shape_code, lhs_code))

    def _print_Array(self,expr):
        lhs_code   = self._print(expr.lhs)

        if len(expr.shape)>1:
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
            st= ','.join(','.join(str(i) for i in array) for array in expr.rhs)
            reshape = True
        else:
            shape_code = '0:' + self._print(expr.shape[0]) + '-1'
            st=','.join(str(i) for i in expr.rhs)
            reshape = False
        shape=','.join(str(i) for i in expr.shape)

        code  = 'allocate({0}({1}))'.format(lhs_code, shape_code)
        code += '\n'
        if reshape:
            code += '{0} = reshape((/{1}/),(/{2}/))'.format(lhs_code, st, str(shape))
        else:
            code += '{0} = (/{1}/)'.format(lhs_code, st)
        return code

    def _print_ZerosLike(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        if isinstance(expr.rhs, IndexedElement):
            shape = []
            for i in expr.rhs.indices:
                if isinstance(i, Slice):
                    shape.append(i)
            rank = len(shape)
        else:
            rank = expr.rhs.rank
        rs = []
        for i in range(1, rank+1):
            l = 'lbound({0},{1})'.format(rhs, str(i))
            u = 'ubound({0},{1})'.format(rhs, str(i))
            r = '{0}:{1}'.format(l,u)
            rs.append(r)
        shape = ', '.join(self._print(i) for i in rs)
        code  = 'allocate({0}({1})) ; {0} = 0'.format(lhs, shape)

        return self._get_statement(code)

    def _print_Len(self, expr):
        if isinstance(expr.rhs,list):
            st=','.join([str(i) for i in expr.rhs])
            return self._get_statement('size((/%s/),1)'%(st))
        else:
            return self._get_statement('size(%s,1)'%(expr.rhs))

    def _print_Shape(self, expr):
        code = ''
        for i,a in enumerate(expr.lhs):
            a_str = self._print(a)
            r_str = self._print(expr.rhs)
            code += '{0} = size({1}, {2})\n'.format(a_str, r_str, str(i+1))
        return self._get_statement(code)

    def _print_Min(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'minval({0})'.format(self._print(arg))
        else:
            raise ValueError("Expecting one argument for the moment.")
        return self._get_statement(code)

    def _print_Max(self,expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'maxval({0})'.format(self._print(arg))
        else:
            raise ValueError("Expecting one argument for the moment.")
        return self._get_statement(code)

    def _print_Dot(self,expr):
        return self._get_statement('dot_product(%s,%s)'%(self._print(expr.expr_l),self._print(expr.expr_r)))

    def _print_Ceil(self,expr):
        return self._get_statement('ceiling(%s)'%(self._print(expr.rhs)))

    def _print_Sign(self,expr):
        # TODO use the appropriate precision from rhs
        return self._get_statement('sign(1.0d0,%s)'%(self._print(expr.rhs)))

    def _print_Declare(self, expr):
        dtype = self._print(expr.dtype)
        # Group the variables by intent

        arg_types        = [type(v) for v in expr.variables]
        arg_ranks        = [v.rank for v in expr.variables]
        arg_allocatables = [v.allocatable for v in expr.variables]

        decs = []
        intent = expr.intent
        vstr = ', '.join(self._print(i.name) for i in expr.variables)

        # TODO ARA improve
        rank        = arg_ranks[0]
        allocatable = arg_allocatables[0]

        # arrays are 0-based in pyccel, to avoid ambiguity with range
        base = '0'
        if allocatable:
            base = ''
        if rank == 0:
            rankstr =  ''
        else:
            rankstr = ', '.join(base+':' for f in range(0, rank))
            rankstr = '(' + rankstr + ')'

        if allocatable:
            allocatablestr = ', allocatable'
        else:
            allocatablestr = ''

        if intent:
            decs.append('{0}, intent({1}) {2} :: {3} {4}'.
                        format(dtype, intent, allocatablestr, vstr, rankstr))
        else:
            decs.append('{0}{1} :: {2} {3}'.
                        format(dtype, allocatablestr, vstr, rankstr))

        return '\n'.join(decs)

    def _print_MPI_comm_world(self, expr):
        return 'MPI_comm_world'

    def _print_MPI_comm_size(self, expr):
        return 'MPI_comm_size'

    def _print_MPI_comm_rank(self, expr):
        return 'MPI_comm_rank'

    def _print_MPI_comm_recv(self, expr):
        return 'MPI_recv'

    def _print_MPI_comm_send(self, expr):
        return 'MPI_send'

    def _print_MPI_comm_irecv(self, expr):
        return 'MPI_irecv'

    def _print_MPI_comm_isend(self, expr):
        return 'MPI_isend'

    def _print_MPI_comm_sendrecv(self, expr):
        return 'MPI_sendrecv'

    def _print_MPI_waitall(self, expr):
        return 'MPI_waitall'

    def _print_MPI_INTEGER(self, expr):
        return 'MPI_INTEGER'

    def _print_MPI_FLOAT(self, expr):
        return 'MPI_FLOAT'

    def _print_MPI_DOUBLE(self, expr):
        return 'MPI_DOUBLE'

    def _print_MPI_status_size(self, expr):
        return 'MPI_status_size'

    def _print_MPI_status_type(self, expr):
        # TODO to remove
        return 'integer, dimension(MPI_STATUS_SIZE)'

    def _print_MPI_proc_null(self, expr):
        return 'MPI_proc_null'

    def _print_MPI_Declare(self, expr):
        dtype = self._print(expr.dtype)
        # Group the variables by intent

        arg_types        = [type(v) for v in expr.variables]
        arg_ranks        = [v.rank for v in expr.variables]
        arg_allocatables = [v.allocatable for v in expr.variables]

        decs = []
        intent = expr.intent
        vstr = ', '.join(self._print(i.name) for i in expr.variables)

        # TODO ARA improve
        rank        = arg_ranks[0]
        allocatable = arg_allocatables[0]

        # arrays are 0-based in pyccel, to avoid ambiguity with range
        base = '0'
        if allocatable:
            base = ''
        if rank == 0:
            rankstr =  ''
        else:
            rankstr = ', '.join(base+':' for f in range(0, rank))
            rankstr = '(' + rankstr + ')'

        if allocatable:
            allocatablestr = ', allocatable'
        else:
            allocatablestr = ''

        if intent:
            decs.append('{0}, intent({1}) {2} :: {3} {4}'.
                        format(dtype, intent, allocatablestr, vstr, rankstr))
        else:
            decs.append('{0}{1} :: {2} {3}'.
                        format(dtype, allocatablestr, vstr, rankstr))

        return '\n'.join(decs)

    def _print_MPI_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        is_procedure = False
        MPI_CONSTANTS = (MPI_comm_world, MPI_status_size, MPI_proc_null)
        if isinstance(expr.rhs, MPI_CONSTANTS):
            rhs_code = self._print(expr.rhs)
            code = '{0} = {1}'.format(lhs_code, rhs_code)
        elif isinstance(expr.rhs, MPI_comm_size):
            rhs_code = self._print(expr.rhs)
            comm = self._print(expr.rhs.comm)
            size = self._print(expr.lhs)
            ierr = self._print(MPI_ERROR)
            code = 'call mpi_comm_size ({0}, {1}, {2})'.format(comm, size, ierr)
        elif isinstance(expr.rhs, MPI_comm_rank):
            rhs_code = self._print(expr.rhs)
            comm = self._print(expr.rhs.comm)
            rank = self._print(expr.lhs)
            ierr = self._print(MPI_ERROR)
            code = 'call mpi_comm_rank ({0}, {1}, {2})'.format(comm, rank, ierr)
        elif isinstance(expr.rhs, MPI_comm_recv):
            rhs_code = self._print(expr.rhs)
            ierr     = self._print(MPI_ERROR)
            istatus  = self._print(MPI_STATUS)

            data   = expr.rhs.data
            count  = expr.rhs.count
            dtype  = expr.rhs.datatype
            source = expr.rhs.source
            tag    = expr.rhs.tag
            comm   = expr.rhs.comm

            args = (data, count, dtype, source, tag, comm, istatus, ierr)
            args  = ', '.join('{0}'.format(self._print(a)) for a in args)
            code = 'call mpi_recv ({0})'.format(args)
        elif isinstance(expr.rhs, MPI_comm_send):
            rhs_code = self._print(expr.rhs)
            ierr     = self._print(MPI_ERROR)

            data  = expr.rhs.data
            count = expr.rhs.count
            dtype = expr.rhs.datatype
            dest  = expr.rhs.dest
            tag   = expr.rhs.tag
            comm  = expr.rhs.comm

            args = (data, count, dtype, dest, tag, comm, ierr)
            t = self._print(data)
            args  = ', '.join('{0}'.format(self._print(a)) for a in args)
            code = 'call mpi_send ({0})'.format(args)
        elif isinstance(expr.rhs, MPI_comm_irecv):
            rhs_code = self._print(expr.rhs)
            ierr     = self._print(MPI_ERROR)

            data    = expr.rhs.data
            count   = expr.rhs.count
            dtype   = expr.rhs.datatype
            source  = expr.rhs.source
            tag     = expr.rhs.tag
            comm    = expr.rhs.comm
            request = expr.rhs.request

            args = (data, count, dtype, source, tag, comm, request, ierr)
            args  = ', '.join('{0}'.format(self._print(a)) for a in args)
            code = 'call mpi_irecv ({0})'.format(args)
        elif isinstance(expr.rhs, MPI_comm_isend):
            rhs_code = self._print(expr.rhs)
            ierr     = self._print(MPI_ERROR)

            data    = expr.rhs.data
            count   = expr.rhs.count
            dtype   = expr.rhs.datatype
            dest    = expr.rhs.dest
            tag     = expr.rhs.tag
            comm    = expr.rhs.comm
            request = expr.rhs.request

            args = (data, count, dtype, dest, tag, comm, request, ierr)
            args  = ', '.join('{0}'.format(self._print(a)) for a in args)
            code = 'call mpi_isend ({0})'.format(args)
        elif isinstance(expr.rhs, MPI_comm_sendrecv):
            rhs_code  = self._print(expr.rhs)
            ierr      = self._print(MPI_ERROR)
            istatus   = self._print(MPI_STATUS)

            senddata  = expr.rhs.senddata
            recvdata  = expr.rhs.recvdata
            sendcount = expr.rhs.sendcount
            recvcount = expr.rhs.recvcount
            sendtype  = expr.rhs.senddatatype
            recvtype  = expr.rhs.recvdatatype
            dest      = expr.rhs.dest
            source    = expr.rhs.source
            sendtag   = expr.rhs.sendtag
            recvtag   = expr.rhs.recvtag
            comm      = expr.rhs.comm

            args = (senddata, sendcount, sendtype, dest,   sendtag, \
                    recvdata, recvcount, recvtype, source, recvtag, \
                    comm, istatus, ierr)
            args  = ', '.join('{0}'.format(self._print(a)) for a in args)
            code = 'call mpi_sendrecv ({0})'.format(args)
        elif isinstance(expr.rhs, MPI_waitall):
            rhs_code = self._print(expr.rhs)
            ierr     = self._print(MPI_ERROR)

            requests = expr.rhs.requests
            count    = expr.rhs.count
            status   = expr.rhs.status

            args = (count, requests, status, ierr)
            args  = ', '.join('{0}'.format(self._print(a)) for a in args)
            code = 'call mpi_waitall ({0})'.format(args)
        else:
            raise TypeError('{0} Not yet implemented.'.format(type(expr.rhs)))
        return self._get_statement(code)



    def _print_NativeBool(self, expr):
        return 'logical'

    def _print_NativeInteger(self, expr):
        return 'integer'

    def _print_NativeFloat(self, expr):
        return 'real'

    def _print_NativeDouble(self, expr):
        return 'real(kind=8)'

    def _print_NativeComplex(self, expr):
        # TODO add precision
        return 'complex(kind=8)'

    def _print_DataType(self, expr):
        name = expr.__class__.__name__
        name = name.split('Pyccel')[-1]
        return 'class({0})'.format(name)

    def _print_EqualityStmt(self, expr):
        return '{0} == {1} '.format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_NotequalStmt(self, expr):
        return '{0} /= {1} '.format(self._print(expr.lhs), self._print(expr.rhs))
    def _print_LOrEq(self, expr):
        return '{0} <= {1} '.format(self._print(expr.lhs), self._print(expr.rhs))
    def _print_Lthan(self, expr):
        return '{0} < {1} '.format(self._print(expr.lhs), self._print(expr.rhs))
    def _print_GOrEq(self, expr):
        return '{0} >= {1} '.format(self._print(expr.lhs), self._print(expr.rhs))
    def _print_Gter(self, expr):
        return '{0} > {1} '.format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_FunctionDef(self, expr):
        name = str(expr.name)
        if expr.cls_name:
            if name in _default_methods:
                name = _default_methods[name]
            name = '{0}_{1}'.format(expr.cls_name, name)
        out_args = []
        decs = []
        body = expr.body
        func_end  = ''
        if len(expr.results) == 1:
            result = expr.results[0]

            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    if not(str(stmt.variables[0].name) == str(result.name)):
                        decs.append(stmt)
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)

            ret_type = self._print(result.dtype)
            func_type = 'function'

            if result.allocatable or (result.rank > 0):
                sig = 'function {0}'.format(name)
                var = Variable(result.dtype, result.name, \
                             rank=result.rank, \
                             allocatable=True, \
                             shape=result.shape)

                dec = Declare(result.dtype, var)
                decs.append(dec)
            else:
                sig = '{0} function {1}'.format(ret_type, name)
                func_end  = ' result({0})'.format(result.name)
        elif len(expr.results) > 1:
            # TODO compute intent
            out_args = expr.results
            for result in expr.results:
                dec = Declare(result.dtype, result, intent='out')
                decs.append(dec)
            sig = 'subroutine ' + name
            func_type = 'subroutine'

            names = [str(res.name) for res in expr.results]
            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    nm = str(stmt.variables[0].name)
                    if not(nm in names):
                        decs.append(stmt)
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)
        else:
            # TODO remove this part
            for result in expr.results:
                arg = Variable(result.dtype, result.name, \
                                  rank=result.rank, \
                                  allocatable=result.allocatable, \
                                  shape=result.shape)

                out_args.append(arg)

                dec = Declare(result.dtype, arg)
                decs.append(dec)
            sig = 'subroutine ' + name
            func_type = 'subroutine'

            names = [str(res.name) for res in expr.results]
            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    nm = str(stmt.variables[0].name)
                    if not(nm in names):
                        decs.append(stmt)
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)
#        else:
#            sig = 'subroutine ' + name
#            func_type = 'subroutine'
        out_code  = ', '.join(self._print(i) for i in out_args)

        arg_code  = ', '.join(self._print(i) for i in expr.arguments)
        if len(out_code) > 0:
            arg_code  = ', '.join(i for i in [arg_code, out_code])

        body_code = '\n'.join(self._print(i) for i in body)
        prelude   = '\n'.join(self._print(i) for i in decs)

        body_code = prelude + '\n\n' + body_code

        return ('{0}({1}) {2}\n'
                'implicit none\n'
#                'integer, parameter:: dp=kind(0.d0)\n'
                '{3}\n\n'
                'end {4}').format(sig, arg_code, func_end, body_code, func_type)

    def _print_Return(self, expr):
        return 'return'

    def _print_Del(self, expr):
        # TODO: treate class case
        variables = '\n'.join('deallocate({})'.format(self._print(i)) for i in expr.variables)
        code = '{}'.format(variables)
        return code

    def _print_ClassDef(self, expr):
        name = expr.name
        base = None # TODO: add base in ClassDef

        decs = '\n'.join(self._print(Declare(i.dtype, i)) for i in expr.attributs)

        aliases = []
        names   = []
        ls = [self._print(i.name) for i in expr.methods]
        for i in ls:
            j = i
            if i in _default_methods:
                j = _default_methods[i]
            aliases.append(j)
            names.append('{0}_{1}'.format(name, self._print(j)))
        methods = '\n'.join('procedure :: {0} => {1}'.format(i,j) for i,j in
                            zip(aliases, names))

        options = ', '.join(i for i in expr.options)

        sig = 'type, {0}'.format(options)
        if not(base is None):
            sig = '{0}, extends({1})'.format(sig, base)

        code = ('{0} :: {1}').format(sig, name)
        if len(decs) > 0:
            code = ('{0}\n'
                    '{1}').format(code, decs)
        if len(methods) > 0:
            code = ('{0}\n'
                    'contains\n'
                    '{1}').format(code, methods)
        code = ('{0}\n'
                'end type {1}').format(code, name)

        methods = '\n\n'.join(self._print(i) for i in expr.methods)
        if len(methods) > 0:
            code = ('{0}\n\n'
                    'contains\n\n'
                    '{1}').format(code, methods)

        return code

    def _print_Break(self,expr):
        return 'Exit'

    def _print_AugAssign(self, expr):
        raise NotImplementedError('Fortran does not support AugAssign')

    def _print_MultiAssign(self, expr):
        if not isinstance(expr.rhs, Function):
            raise TypeError('Expecting a Function call.')
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.rhs.args]
        func = expr.rhs.func
        args    = ', '.join(self._print(i) for i in args)
        outputs = ', '.join(self._print(i) for i in expr.lhs)
        return 'call {0} ({1}, {2})'.format(func, args, outputs)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        # decrement stop by 1 because of the python convention
        stop = stop - 1

        body = '\n'.join(self._print(i) for i in expr.body)
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop,
                        step=step, body=body)

    def _print_While(self,expr):
        body = '\n'.join(self._print(i) for i in expr.body)
        return ('do while ({test}) \n'
                '{body}\n'
                'end do').format(test=self._print(expr.test),body=body)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) then" % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines.append("else")
            else:
                lines.append("else if (%s) then" % self._print(c))
            if isinstance(e, list):
                for ee in e:
                    lines.append(self._print(ee))
            else:
                lines.append(self._print(e))
        lines.append("end if")
        return "\n".join(lines)

    def _print_MatrixElement(self, expr):
        return "{0}({1}, {2})".format(expr.parent, expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if len(pure_imaginary) > 0:
            if len(mixed) > 0:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_FunctionCall(self, expr):
        # for the moment, this is only used if the function has not arguments
        func = expr.func
        name = self._print(func.name)
        code_args = ''
        if not(func.arguments) is None:
            code_args = ', '.join(sstr(i) for i in func.arguments)
        return '{0}({1})'.format(name, code_args)

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp == -1:
            one = Float(1.0)
            code = '{0}/{1}'.format(self._print(one), \
                                    self.parenthesize(expr.base, PREC))
            return code
        elif expr.exp == 0.5:
            if expr.base.is_integer:
                # Fortan intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_IndexedVariable(self, expr):
        return "{0}".format(str(expr))

    def _print_IndexedElement(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Slice(self, expr):
        if expr.start is None:
            start = ''
        else:
            start = self._print(expr.start)
        if expr.end is None:
            end = ''
        else:
            end = expr.end - 1
            end = self._print(end)
        return '{0} : {1}'.format(start, end)

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append("! " + line[1:].lstrip())
            else:
                result.append(line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos
        # split line by line and add the splitted lines to result
        result = []
        trailing = ' &'
        for line in lines:
            if line.startswith("! "):
                # comment line
                if len(line) > 72:
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while len(line) > 0:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % ("! ", hunk))
                else:
                    result.append(line)
            else:
                # code line
                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while len(line) > 0:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s" % ("      " , hunk))
        return result

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [ line.lstrip(' \t') for line in code ]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n', \
                       'else', 'type', 'subroutine', 'function')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif', \
                       'else', 'endtype', 'end type', \
                       'endfunction', 'end function', \
                       'endsubroutine', 'end subroutine')

        increase = [ int(any(map(line.startswith, inc_keyword)))
                     for line in code ]
        decrease = [ int(any(map(line.startswith, dec_keyword)))
                     for line in code ]
        continuation = [ int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code ]

        level = 0
        cont_padding = 0
        tabwidth = self._default_settings['tabwidth']
        new_code = []
        for i, line in enumerate(code):
            if line == '' or line == '\n':
                new_code.append(line)
                continue
            level -= decrease[i]

            padding = " "*(level*tabwidth + cont_padding)

            line = "%s%s" % (padding, line)

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        return new_code


def fcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of c code

    expr : Expr
        A sympy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    """

    return FCodePrinter(settings).doprint(expr, assign_to)
