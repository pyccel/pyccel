# coding: utf-8

"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""

from __future__ import print_function, division
import string
from itertools import groupby
import numpy as np

from sympy.core import Symbol
from sympy.core import Float
from sympy.core import S, Add, N
from sympy.core import Tuple
from sympy.core.function import Function
from sympy.core.compatibility import string_types
from sympy.printing.precedence import precedence
from sympy import Eq,Ne,true,false

from sympy.utilities.iterables import iterable
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse

from pyccel.types.ast import AddOp, MulOp, SubOp, DivOp
from pyccel.types.ast import FunctionDef
from pyccel.types.ast import FunctionCall
from pyccel.types.ast import ZerosLike
from pyccel.types.ast import NativeBool, NativeFloat
from pyccel.types.ast import NativeComplex, NativeDouble, NativeInteger
from pyccel.types.ast import Range, Tensor, Block
from pyccel.types.ast import (Assign, MultiAssign, AugAssign, \
                              Result, \
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
from pyccel.parallel.mpi import MPI_comm_sendrecv_replace
from pyccel.parallel.mpi import MPI_comm_barrier
from pyccel.parallel.mpi import MPI_comm_bcast
from pyccel.parallel.mpi import MPI_waitall
from pyccel.parallel.mpi import MPI_comm_scatter
from pyccel.parallel.mpi import MPI_comm_gather
from pyccel.parallel.mpi import MPI_comm_allgather
from pyccel.parallel.mpi import MPI_comm_alltoall
from pyccel.parallel.mpi import MPI_comm_reduce
from pyccel.parallel.mpi import MPI_comm_allreduce
from pyccel.parallel.mpi import MPI_comm_split
from pyccel.parallel.mpi import MPI_comm_free
from pyccel.parallel.mpi import MPI_comm_cart_create
from pyccel.parallel.mpi import MPI_comm_cart_coords
from pyccel.parallel.mpi import MPI_comm_cart_shift
from pyccel.parallel.mpi import MPI_comm_cart_sub
from pyccel.parallel.mpi import MPI_dims_create
from pyccel.parallel.mpi import MPI_SUM, MPI_PROD
from pyccel.parallel.mpi import MPI_MIN, MPI_MAX
from pyccel.parallel.mpi import MPI_Tensor
from pyccel.parallel.mpi import MPI_TensorCommunication

from pyccel.clapp.plaf import Matrix_dns
from pyccel.clapp.plaf import Matrix_dns_create


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

    def _print_EmptyLine(self, expr):
        return '\n'

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

    def _print_DottedName(self, expr):
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

        if expr.grid is None:
            if isinstance(expr.shape, Tuple):
                # this is a correction. problem on LRZ
                shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
            elif isinstance(expr.shape, str):
                shape_code = '0:' + self._print(expr.shape) + '-1'
            else:
                raise TypeError('Unknown type of shape'+str(type(expr.shape)))

            if not isinstance(expr.lhs, Variable):
                raise TypeError('Expecting lhs to be a Variable')
        else:
            # TODO check tensor type
            #      this only works with steps = 1
            tensor = expr.grid
            if isinstance(tensor, Tensor):
                starts = [r.start for r in tensor.ranges]
                ends   = [r.stop  for r in tensor.ranges]
                steps  = [r.step  for r in tensor.ranges]
            elif isinstance(tensor, Range):
                starts = [tensor.start]
                ends   = [tensor.stop ]
                steps  = [tensor.step ]

            if isinstance(tensor, MPI_Tensor):
                pads = tensor.pads
            else:
                pads = np.zeros(2, dtype=int)

            shape_code = ', '.join('{0}:{1}'.format(self._print(s-p),  \
                                                    self._print(e+p)) \
                                   for (s,e, p) in zip(starts, ends, pads))

        init_value = None
        dtype = expr.lhs.dtype
        init_value = expr.init_value

        code_alloc = "allocate({0}({1}))".format(lhs_code, shape_code)
        code_init = "{0} = {1}".format(lhs_code, self._print(init_value))
        code = "{0}; {1}".format(code_alloc, code_init)
        return self._get_statement(code)

    def _print_Array(self,expr):
        lhs_code   = self._print(expr.lhs)

        if len(expr.shape)>1:
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
            st= ','.join(','.join(self._print(i) for i in array) for array in expr.rhs)
            reshape = True
        else:
            shape_code = '0:' + self._print(expr.shape[0]) + '-1'
            st=','.join(self._print(i) for i in expr.rhs)
            reshape = False
        shape=','.join(self._print(i) for i in expr.shape)

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
        ierr     = self._print(MPI_ERROR)
        istatus  = self._print(MPI_STATUS)

        data   = expr.data
        count  = expr.count
        dtype  = expr.datatype
        source = expr.source
        tag    = expr.tag
        comm   = expr.comm

        args = (data, count, dtype, source, tag, comm, istatus, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_recv ({0})'.format(args)
        return code

    def _print_MPI_comm_send(self, expr):
        ierr  = self._print(MPI_ERROR)

        data  = expr.data
        count = expr.count
        dtype = expr.datatype
        dest  = expr.dest
        tag   = expr.tag
        comm  = expr.comm

        args = (data, count, dtype, dest, tag, comm, ierr)
        t = self._print(data)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_send ({0})'.format(args)
        return code

    def _print_MPI_comm_irecv(self, expr):
        ierr     = self._print(MPI_ERROR)

        data    = expr.data
        count   = expr.count
        dtype   = expr.datatype
        source  = expr.source
        tag     = expr.tag
        comm    = expr.comm
        request = expr.request

        args = (data, count, dtype, source, tag, comm, request, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_irecv ({0})'.format(args)
        return code

    def _print_MPI_comm_isend(self, expr):
        ierr     = self._print(MPI_ERROR)

        data    = expr.data
        count   = expr.count
        dtype   = expr.datatype
        dest    = expr.dest
        tag     = expr.tag
        comm    = expr.comm
        request = expr.request

        args = (data, count, dtype, dest, tag, comm, request, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_isend ({0})'.format(args)
        return code

    def _print_MPI_comm_sendrecv(self, expr):
        ierr      = self._print(MPI_ERROR)
        istatus   = self._print(MPI_STATUS)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        sendcount = expr.sendcount
        recvcount = expr.recvcount
        sendtype  = expr.senddatatype
        recvtype  = expr.recvdatatype
        dest      = expr.dest
        source    = expr.source
        sendtag   = expr.sendtag
        recvtag   = expr.recvtag
        comm      = expr.comm

        args = (senddata, sendcount, sendtype, dest,   sendtag, \
                recvdata, recvcount, recvtype, source, recvtag, \
                comm, istatus, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_sendrecv ({0})'.format(args)
        return code

    def _print_MPI_comm_sendrecv_replace(self, expr):
        ierr      = self._print(MPI_ERROR)
        istatus   = self._print(MPI_STATUS)

        data      = expr.data
        count     = expr.count
        dtype     = expr.datatype
        dest      = expr.dest
        source    = expr.source
        sendtag   = expr.sendtag
        recvtag   = expr.recvtag
        comm      = expr.comm

        args = (data, count, dtype, dest, sendtag, source, recvtag, \
                comm, istatus, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_sendrecv_replace ({0})'.format(args)
        return code

    def _print_MPI_waitall(self, expr):
        ierr     = self._print(MPI_ERROR)

        requests = expr.requests
        count    = expr.count
        status   = expr.status

        args = (count, requests, status, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_waitall ({0})'.format(args)
        return code

    def _print_MPI_comm_barrier(self, expr):
        comm     = self._print(expr.comm)
        ierr     = self._print(MPI_ERROR)

        code = 'call mpi_barrier ({0}, {1})'.format(comm, ierr)
        return code

    def _print_MPI_comm_bcast(self, expr):
        ierr      = self._print(MPI_ERROR)

        data      = expr.data
        count     = expr.count
        dtype     = expr.datatype
        root      = expr.root
        comm      = expr.comm

        args = (data, count, dtype, root, comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_bcast ({0})'.format(args)
        return code

    def _print_MPI_comm_scatter(self, expr):
        ierr      = self._print(MPI_ERROR)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        sendcount = expr.sendcount
        recvcount = expr.recvcount
        sendtype  = expr.senddatatype
        recvtype  = expr.recvdatatype
        root      = expr.root
        comm      = expr.comm

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                root, comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_scatter ({0})'.format(args)
        return code

    def _print_MPI_comm_gather(self, expr):
        ierr      = self._print(MPI_ERROR)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        sendcount = expr.sendcount
        recvcount = expr.recvcount
        sendtype  = expr.senddatatype
        recvtype  = expr.recvdatatype
        root      = expr.root
        comm      = expr.comm

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                root, comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_gather ({0})'.format(args)
        return code

    def _print_MPI_comm_allgather(self, expr):
        ierr      = self._print(MPI_ERROR)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        sendcount = expr.sendcount
        recvcount = expr.recvcount
        sendtype  = expr.senddatatype
        recvtype  = expr.recvdatatype
        comm      = expr.comm

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_allgather ({0})'.format(args)
        return code

    def _print_MPI_comm_alltoall(self, expr):
        ierr      = self._print(MPI_ERROR)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        sendcount = expr.sendcount
        recvcount = expr.recvcount
        sendtype  = expr.senddatatype
        recvtype  = expr.recvdatatype
        comm      = expr.comm

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_alltoall ({0})'.format(args)
        return code

    def _print_MPI_comm_reduce(self, expr):
        ierr      = self._print(MPI_ERROR)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        count     = expr.count
        dtype     = expr.datatype
        op        = expr.op
        root      = expr.root
        comm      = expr.comm

        args = (senddata, recvdata, count, dtype, op, \
                root, comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_reduce ({0})'.format(args)
        return code

    def _print_MPI_comm_allreduce(self, expr):
        ierr      = self._print(MPI_ERROR)

        senddata  = expr.senddata
        recvdata  = expr.recvdata
        count     = expr.count
        dtype     = expr.datatype
        op        = expr.op
        comm      = expr.comm

        args = (senddata, recvdata, count, dtype, op, \
                comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_allreduce ({0})'.format(args)
        return code

    def _print_MPI_comm_split(self, expr):
        ierr      = self._print(MPI_ERROR)

        color     = expr.color
        key       = expr.key
        comm      = expr.comm
        newcomm   = expr.newcomm

        args = (comm, color, key, newcomm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_comm_split ({0})'.format(args)
        return code

    def _print_MPI_comm_free(self, expr):
        ierr = self._print(MPI_ERROR)
        comm = expr.comm

        args = (comm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_comm_free ({0})'.format(args)
        return code

    def _print_MPI_comm_cart_create(self, expr):
        ierr      = self._print(MPI_ERROR)

        ndims     = expr.ndims
        dims      = expr.dims
        periods   = expr.periods
        reorder   = expr.reorder
        comm      = expr.comm
        newcomm   = expr.newcomm

        args = (comm, ndims, dims, periods, reorder, newcomm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_cart_create ({0})'.format(args)
        return code

    def _print_MPI_comm_cart_coords(self, expr):
        ierr      = self._print(MPI_ERROR)

        rank   = expr.rank
        ndims  = expr.ndims
        coords = expr.coords
        comm   = expr.comm

        args = (comm, rank, ndims, coords, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_cart_coords ({0})'.format(args)
        return code

    def _print_MPI_comm_cart_shift(self, expr):
        ierr      = self._print(MPI_ERROR)

        direction = expr.direction
        disp      = expr.disp
        source    = expr.source
        dest      = expr.dest
        comm      = expr.comm

        args = (comm, direction, disp, source, dest, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_cart_shift ({0})'.format(args)
        return code

    def _print_MPI_comm_cart_sub(self, expr):
        ierr      = self._print(MPI_ERROR)

        dims      = expr.dims
        comm      = expr.comm
        newcomm   = expr.newcomm

        args = (comm, dims, newcomm, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_cart_sub ({0})'.format(args)
        return code

    def _print_MPI_type_contiguous(self, expr):
        ierr     = self._print(MPI_ERROR)

        count       = expr.count
        oldtype     = expr.oldtype
        newtype     = expr.newtype

        args = (count, oldtype, newtype, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call MPI_type_contiguous ({0})'.format(args)

        commit = 'call MPI_type_commit ({0}, {1})'.format(newtype, ierr)

        code = '{0}\n{1}'.format(code, commit)
        return code

    def _print_MPI_type_vector(self, expr):
        ierr     = self._print(MPI_ERROR)

        count       = expr.count
        blocklength = expr.blocklength
        stride      = expr.stride
        oldtype     = expr.oldtype
        newtype     = expr.newtype

        args = (count, blocklength, stride, oldtype, newtype, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call MPI_type_vector ({0})'.format(args)

        commit = 'call MPI_type_commit ({0}, {1})'.format(newtype, ierr)

        code = '{0}\n{1}'.format(code, commit)
        return code

    def _print_MPI_dims_create(self, expr):
        ierr      = self._print(MPI_ERROR)

        nnodes  = expr.nnodes
        ndims   = expr.ndims
        dims    = expr.dims

        args = (nnodes, ndims, dims, ierr)
        args  = ', '.join('{0}'.format(self._print(a)) for a in args)
        code = 'call mpi_dims_create ({0})'.format(args)
        return code

    def _print_MPI_INTEGER(self, expr):
        return 'MPI_INTEGER'

    def _print_MPI_REAL(self, expr):
        return 'MPI_REAL'

    def _print_MPI_DOUBLE(self, expr):
        return 'MPI_DOUBLE'

    def _print_MPI_SUM(self, expr):
        return 'MPI_SUM'

    def _print_MPI_PROD(self, expr):
        return 'MPI_PROD'

    def _print_MPI_MAX(self, expr):
        return 'MPI_MAX'

    def _print_MPI_MIN(self, expr):
        return 'MPI_MIN'

    def _print_MPI_status_size(self, expr):
        return 'MPI_status_size'

    def _print_MPI_status_type(self, expr):
        # TODO to remove
        return 'integer, dimension(MPI_STATUS_SIZE)'

    def _print_MPI_proc_null(self, expr):
        return 'MPI_proc_null'

    def _print_MPI_comm(self, expr):
        return expr.name

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

    def _print_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        is_procedure = False
        if isinstance(expr.rhs, FunctionDef):
            rhs_code = self._print(expr.rhs.name)
            is_procedure = (expr.rhs.kind == 'procedure')
        elif isinstance(expr.rhs, FunctionCall):
            rhs_code = self._print(expr.rhs.name)
            cls_name = expr.rhs.func.cls_name
            if cls_name:
                rhs_code = '{0} % {1}'.format(lhs_code, rhs_code)
            is_procedure = (expr.rhs.kind == 'procedure')
        else:
            rhs_code = self._print(expr.rhs)

        code = ''
        if (expr.status == 'unallocated') and not (expr.like is None):
            stmt = ZerosLike(lhs_code, expr.like)
            code += self._print(stmt)
            code += '\n'
        if not is_procedure:
            code += '{0} = {1}'.format(lhs_code, rhs_code)
        else:
            code_args = ''
            func = expr.rhs
            # func here is of instance FunctionCall
            cls_name = func.func.cls_name
            if (not cls_name) and \
               (not func.arguments is None) and \
               (len(func.arguments) > 0):
                code_args = ', '.join(self._print(i) for i in func.arguments)
                code_args = '{0}, {1}'.format(code_args, lhs_code)
            else:
                code_args = ', '.join(self._print(i) for i in func.arguments)
#                code_args = lhs_code
            code = 'call {0}({1})'.format(rhs_code, code_args)
        return self._get_statement(code)

    def _print_MPI_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        is_procedure = False
        MPI_CONSTANTS = (MPI_comm_world, MPI_status_size, MPI_proc_null)
        if isinstance(expr.rhs, MPI_CONSTANTS):
            rhs_code = self._print(expr.rhs)
            code = '{0} = {1}'.format(lhs_code, rhs_code)
        elif isinstance(expr.rhs, MPI_comm_size):
            comm = self._print(expr.rhs.comm)
            size = self._print(expr.lhs)
            ierr = self._print(MPI_ERROR)
            code = 'call mpi_comm_size ({0}, {1}, {2})'.format(comm, size, ierr)
        elif isinstance(expr.rhs, MPI_comm_rank):
            comm = self._print(expr.rhs.comm)
            rank = self._print(expr.lhs)
            ierr = self._print(MPI_ERROR)
            code = 'call mpi_comm_rank ({0}, {1}, {2})'.format(comm, rank, ierr)
        elif isinstance(expr.rhs, MPI):
            code = self._print(expr.rhs)
        else:
            raise TypeError('{0} Not yet implemented.'.format(type(expr.rhs)))
        return self._get_statement(code)


    def _print_Sync(self, expr):
        return 'Sync'

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

    def _print_BooleanTrue(self, expr):
        return '.true.'

    def _print_BooleanFalse(self, expr):
        return '.false.'

    def _print_DataType(self, expr):
        c_name = expr.__class__.__name__

        name = c_name.split('Pyccel')[-1]
        if len(name) > 0:
            cls = eval('{0}'.format(name))()
            name = str(cls.dtype)
        else:
            raise NotImplementedError('Only Pyccel DataType is available')

        return 'type({0})'.format(name)

    def _print_Equality(self, expr):
        return '{0} == {1} '.format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_Unequality(self, expr):
        return '{0} /= {1} '.format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_BooleanTrue(self, expr):
        return '.True.'

    def _print_BooleanFalse(self,expr):
        return '.False.'

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
            result,val_result = expr.results[0]


            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    if not(str(stmt.variables[0].name) == str(result.name)):
                        decs.append(stmt)
                elif isinstance(stmt,Result):
                    for i,j in stmt.result_variables:
                        if not j==None:
                            body.append(Assign(i,j))
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
            out_args = [result for result,val_result in expr.results]
            for result,val_result in expr.results:
                if result in expr.arguments:
                    dec = Declare(result.dtype, result, intent='inout')
                else:
                    dec = Declare(result.dtype, result, intent='out')
                decs.append(dec)
            sig = 'subroutine ' + name
            func_type = 'subroutine'

            names = [str(res.name) for res,i in expr.results]
            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    nm = str(stmt.variables[0].name)
                    if not(nm in names):
                        decs.append(stmt)
                elif isinstance(stmt,Result):
                    for i,j in stmt.result_variables:
                        if not j==None:
                            body.append(Assign(i,j))
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)
        else:
            # TODO remove this part
            for result,val_result in expr.results:
                arg = Variable(result.dtype, result.name, \
                                  rank=result.rank, \
                                  allocatable=result.allocatable, \
                                  shape=result.shape)

                out_args.append(arg)

                dec = Declare(result.dtype, arg)
                decs.append(dec)
            sig = 'subroutine ' + name
            func_type = 'subroutine'

            names = [str(res.name) for res,i in expr.results]
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
        #remove parametres intent(inout) from out_args to prevent repetition
        for i in expr.arguments:
            if i in out_args:
                out_args.remove(i)

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
        code = ''
        for var in expr.variables:
            if isinstance(var, Variable):
                code = 'deallocate({0}){1}'.format(self._print(var), code)
            elif isinstance(var, MPI_Tensor):
                stmts = var.free_statements()
                for stmt in stmts:
                    code = '{0}\n{1}'.format(self._print(stmt), code)
            else:
                msg  = 'Only Variable and MPI_Tensor are treated.'
                msg += ' Given {0}'.format(type(var))
                raise NotImplementedError(msg)
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
        return 'exit'

    def _print_Continue(self,expr):
        return 'continue'

    def _print_AugAssign(self, expr):
        lhs    = expr.lhs
        op     = expr.op
        rhs    = expr.rhs
        strict = expr.strict
        status = expr.status
        like   = expr.like

        if isinstance(op, AddOp):
            rhs = lhs + rhs
        elif isinstance(op, MulOp):
            rhs = lhs * rhs
        elif isinstance(op, SubOp):
            rhs = lhs - rhs
        # TODO fix bug with division of integers
        elif isinstance(op, DivOp):
            rhs = lhs / rhs
        else:
            raise ValueError('Unrecongnized operation', op)

        stmt = Assign(lhs, rhs, strict=strict, status=status, like=like)
        return self._print(stmt)

    def _print_MultiAssign(self, expr):
        if not isinstance(expr.rhs, Function):
            raise TypeError('Expecting a Function call.')
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.rhs.args]
        func = expr.rhs.func
        args    = ', '.join(self._print(i) for i in args)
        outputs = ', '.join(self._print(i) for i in expr.lhs)
        return 'call {0} ({1}, {2})'.format(func, args, outputs)

    def _print_Range(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop-1)
        step  = self._print(expr.step)
        return '{0}, {1}, {2}'.format(start, stop, step)

    def _print_Tile(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        return '{0}, {1}'.format(start, stop)

    # TODO iterators
    def _print_For(self, expr):
        prolog = ''
        epilog = ''

        # ...
        def _do_range(target, iter, prolog, epilog):
            if not isinstance(iter, Range):
                msg = "Only iterable currently supported is Range"
                raise NotImplementedError(msg)

            tar        = self._print(target)
            range_code = self._print(iter)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog += 'end do\n'

            return prolog, epilog
        # ...

        # ...
        def _iprint(i):
            if isinstance(i, Block):
                _prelude, _body = self._print_Block(i)
                return '{0}'.format(_body)
            else:
                return '{0}'.format(self._print(i))
        # ...

        if not isinstance(expr.iterable, (Range, Tensor, MPI_Tensor)):
            msg  = "Only iterable currently supported are Range, "
            msg += "Tensor and MPI_Tensor"
            raise NotImplementedError(msg)

        if isinstance(expr.iterable, Range):
            prolog, epilog = _do_range(expr.target[0], expr.iterable, \
                                       prolog, epilog)
        elif isinstance(expr.iterable, (Tensor, MPI_Tensor)):
            for i,a in zip(expr.target, expr.iterable.ranges):
                prolog, epilog = _do_range(i, a, \
                                           prolog, epilog)

        body = '\n'.join(_iprint(i) for i in expr.body)

        return ('{prolog}'
                '{body}\n'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)

    def _print_Block(self, expr):
        body    = '\n'.join(self._print(i) for i in expr.body)
        prelude = '\n'.join(self._print(i) for i in expr.declarations)
        return prelude, body

    def _print_While(self,expr):
        body = '\n'.join(self._print(i) for i in expr.body)
        return ('do while ({test}) \n'
                '{body}\n'
                'end do').format(test=self._print(expr.test),body=body)

    def _print_If(self, expr):
        # ...
        def _iprint(i):
            if isinstance(i, Block):
                _prelude, _body = self._print_Block(i)
                return '{0}'.format(_body)
            else:
                return '{0}'.format(self._print(i))
        # ...

        lines = []
        for i, (c, e) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) then" % _iprint(c))
            elif i == len(expr.args) - 1 and c == True:
                lines.append("else")
            else:
                lines.append("else if (%s) then" % _iprint(c))
            if isinstance(e, list):
                for ee in e:
                    lines.append(_iprint(ee))
            else:
                lines.append(_iprint(e))
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
