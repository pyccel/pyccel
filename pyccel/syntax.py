# coding: utf-8
from sympy import Symbol, sympify, Integer, Float, Add, Mul
from sympy import true, false,pi
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.core.basic import Basic
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.power import Pow
from sympy.core.function import Function
from sympy import preorder_traversal
from sympy import Abs,sqrt,sin,cos,exp,log,sign,csc, cos, \
              sec, tan, cot, asin, acsc, acos, asec, atan, acot, atan2,factorial

from pyccel.types.ast import (For, Assign, Declare, Variable, \
                              datatype, While, NativeFloat, \
                              EqualityStmt, NotequalStmt, \
                              Argument, InArgument, InOutArgument, \
                              MultiAssign, OutArgument, Result, \
                              FunctionDef, Import, Print, \
                              Comment, AnnotatedComment, \
                              IndexedVariable, Slice, If, \
                              ThreadID, ThreadsNumber, \
                              Rational, NumpyZeros, NumpyLinspace, \
                              NumpyOnes, NumpyArray, LEN, Dot, Min, Max,
    IndexedElement)

DEBUG = False
#DEBUG = True

# TODO: 1. check that every stmt is well implementing
#          the local_vars and stmt_vars properties.

__all__ = ["Pyccel", \
           "Expression", "Term", "Operand", \
           "FactorSigned", "FactorUnary", "FactorBinary", \
           # statements
           "AssignStmt", "MultiAssignStmt", "DeclarationStmt", \
           # compound stmts
           "ForStmt", "IfStmt", "SuiteStmt", \
           # Flow statements
           "FlowStmt", "BreakStmt", "ContinueStmt", \
           "RaiseStmt", "YieldStmt", "ReturnStmt", \
           "DelStmt", "PassStmt", "FunctionDefStmt", \
           "ImportFromStmt", \
           "ConstructorStmt", \
           "CommentStmt", \
           # Multi-threading
           "ThreadStmt", \
           # python standard library statements
           "PythonPrintStmt", \
           # numpy statments
           "NumpyZerosStmt", "NumpyZerosLikeStmt", \
           "NumpyOnesStmt", "NumpyLinspaceStmt", \
           # Test
           "Test", "OrTest", "AndTest", "NotTest", "Comparison", \
           # Trailers
           "Trailer", "TrailerArgList", "TrailerSubscriptList", \
           "TrailerSlice", "TrailerSliceRight", "TrailerSliceLeft"
           ]


# Global variable namespace
namespace    = {}
stack        = {}
settings     = {}
variables    = {}
declarations = {}

operators = {}

namespace["True"]  = true
namespace["False"] = false
namespace["pi"]=pi

def Check_type(var_name,expr):
    datatype='int'
    rank=0
    allocatable=False
    shape=[]
    s=[]
    def pre(expr):


        if(type(expr)==Indexed) or type(expr)==IndexedElement:
            s.append((expr.args[0],expr.args[1]))
            return


        elif len(expr.args)==0:
            s.append(expr)
        for arg in expr.args:
            pre(arg)

    pre(expr.expr)

    if isinstance(expr,Expression):
        for i in s:
            if isinstance(i,tuple):
                if isinstance(i[0],IndexedBase)and isinstance(i[1],Slice):
                     if variables[str(i[0])].dtype=='float':
                         datatype='float'
                     if  variables[str(i[0])].allocatable:
                         allocatable=True


                     if not variables[str(i[0])].shape==None:
                            temp1=variables[str(i[0])].shape
                            if(isinstance(temp1,tuple)):
                                rank=len(temp1)
                                if all(i[k].start>=0 and i[k].end<=temp1[k-1] for k in range(1,rank+1)):
                                   shape.append(tuple([i[k].end-i[k].start for k in range(1,rank+1)]))
                                else:
                                    raise TypeError('dimension mismatch')
                            elif isinstance(temp1,int):
                                if i[1].start>=0 and i[1].end<=temp1:
                                    shape.append(i[1].end-i[1].start)
                                else:
                                    raise TypeError('dimension mismatch')
                            else:
                                raise TypeError('shape must be an int or a tuple of int')
                     else:
                         raise TypeError('dimension mismatch')
                elif isinstance(i[0],IndexedBase)and i[1].is_integer:
                    datatype=variables[str(i[0])].dtype
            elif isinstance(i,Symbol):

                if isinstance(variables[str(i)].dtype,NativeFloat):
                    datatype='float'
                if  variables[str(i)].allocatable:
                    allocatable=True
                if not variables[str(i)].shape==None:
                    shape.append(variables[str(i)].shape)
            elif i.is_real and not i.is_integer:
                    datatype='float'
    name=sympify(var_name)
    if len(shape)>0:
        if all(x==shape[0] for x in shape):
            shape=shape[0]

            if isinstance(shape,(tuple,list)):
                shape=tuple(map(int,shape))
                rank=len(shape)
            elif isinstance(shape,int):
                rank=1
            elif isinstance(shape,Symbol):
                if shape.is_integer:
                    rank=1
                    shape=int(shape)
        else:
            raise TypeError('shape are not equal')

    else:
        shape=None

    return {'datatype':datatype,'name':name , 'rank':rank, 'allocatable':allocatable,'shape':shape}

def insert_variable(var_name, \
                    var=None, \
                    datatype=None, \
                    rank=None, \
                    allocatable=None, \
                    shape=None, \
                    is_argument=False):
    if type(var_name) in [int, float]:
        return

    if DEBUG:
        print ">>> trying to insert : ", var_name
        txt = '    datatype={0}, rank={1}, allocatable={2}, shape={3}, is_argument={4}'\
                .format(datatype, rank, allocatable, shape, is_argument)
        print txt

    if var_name in namespace:
        var = variables[var_name]
        if datatype is None:
            datatype = var.dtype
        if rank is None:
            rank = var.rank
        if allocatable is None:
            allocatable = var.allocatable
        if shape is None:
            shape = var.shape
        if isinstance(var, InArgument):
            is_argument = True
    else:
        if datatype is None:
            datatype = 'float'
        if rank is None:
            rank = 0
        if allocatable is None:
            allocatable = False

    is_integer = (datatype == 'int')

    # we first create a sympy symbol
    s = Symbol(var_name, integer=is_integer)

    # we create a variable (for annotation)
    if not is_argument:
        var = Variable(datatype, s, \
                       rank=rank, \
                       allocatable=allocatable, \
                       shape=shape)
    else:
        var = InArgument(datatype, s, \
                         rank=rank, \
                         allocatable=allocatable, \
                         shape=shape)

    # we create a declaration for code generation
    dec = Declare(datatype, var)

    if var_name in namespace:
        namespace.pop(var_name)
        variables.pop(var_name)
        declarations.pop(var_name)

    namespace[var_name]    = s
    variables[var_name]    = var
    declarations[var_name] = dec

# ...
def do_arg(a):
    if isinstance(a, str):
        arg = Symbol(a, integer=True)
    elif isinstance(a, (Integer, Float)):
        arg = a
    elif isinstance(a, Expression):
        arg = a.expr
        if isinstance(arg, Symbol):
            arg = Symbol(arg.name, integer=True)
        else:
            arg = convert_to_integer_expression(arg)
    else:
        raise Exception('Wrong instance in do_arg')

    return arg
# ...

# ...
def is_integer_expression(expr):
    """determines is an expression is an integer expression.
    We check if there is an integer Symbol."""
    for arg in preorder_traversal(expr):
        if isinstance(arg, Symbol):
            if arg.is_integer:
                return True
    return False
# ...

# ...
def convert_to_integer_expression(expr):
    numbers = []
    for arg in preorder_traversal(expr):
        if isinstance(arg, Float):
            numbers.append(arg)
    e = expr
    for n in numbers:
        e = e.subs(n, int(n))
    return e
# ...

# ...
class Pyccel(object):
    """Class for Pyccel syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Pyccel.

        """
        self.statements   = kwargs.pop('statements',   [])

        # ... reset global variables
        namespace    = {}
        stack        = {}
        settings     = {}
        variables    = {}
        declarations = {}

        operators = {}

        namespace["True"]  = true
        namespace["False"] = false
        # ...

    @property
    def declarations(self):
        d = {}
        for key,dec in declarations.items():
            if not(isinstance(dec, Argument)):
                d[key] = dec
        return d

class Number(object):
    """Class representing a number."""
    def __init__(self, **kwargs):
        """
        """
        self.name     = kwargs.pop('name')
        self.datatype = kwargs.pop('datatype')

        namespace[self.name] = self

    @property
    def expr(self):
        return Symbol(self.name)

class BasicStmt(object):
    def __init__(self, **kwargs):
        # TODO declarations and statements must be a dictionary
        self.statements   = []

    @property
    def declarations(self):
        return [declarations[v] for v in self.stmt_vars + self.local_vars]

    @property
    def local_declarations(self):
        return [declarations[v] for v in self.local_vars]

    @property
    def local_vars(self):
        """must be defined byt the statement."""
        return []

    @property
    def stmt_vars(self):
        """must be defined byt the statement."""
        return []

    def update(self):
        pass

#    # TODO move somewhere else
#    def do_trailer(self, trailer):
##        # only slices of the form a:b are possible
##        # this assumes that inputs.args is of length 2
##        if is_slice:
##            assert(len(inputs.args) == 2)
##
##            start = do_arg(inputs.args[0])
##            end   = do_arg(inputs.args[1])
##
##            args = Slice(start, end)
#
#        if isinstance(trailer, Trailer):
#            inputs = trailer.subs
#            if inputs:
#                args = []
#                for a in inputs.args:
#                    arg = do_arg(a)
#
#                    # TODO treat n correctly
#                    n = Symbol('n', integer=True)
#                    i = Idx(arg, n)
#                    args.append(i)
#                return args
#        else:
#            raise Exception('Wrong Trailer type. given {}'.format(type(trailer)))

class ConstructorStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs         = kwargs.pop('lhs')
        self.constructor = kwargs.pop('constructor')

        super(ConstructorStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        """
        var_name    = str(self.lhs)
        constructor = str(self.constructor)
        # TODO improve
        if constructor in ["array_1", "array_2", "array_3"]:
            if constructor == "array_2":
                rank = 2
                datatype = 'float'
            elif constructor == "array_3":
                rank = 3
                datatype = 'float'
            else:
                rank = 1
                datatype = 'float'
        else:
            rank     = 0
            datatype = constructor
        insert_variable(var_name, datatype=datatype, rank=rank)
        return Comment("")

class DeclarationStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.variables_name = kwargs.pop('variables')
        self.datatype = kwargs.pop('datatype')


        self.variables = []
        # TODO create the appropriate type, not only Number
        for var in self.variables_name:
            self.variables.append(Number(name=var, datatype=self.datatype))

        super(DeclarationStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        """
        datatype = str(self.datatype)
        decs = []
        # TODO depending on additional options from the grammar
        for var in self.variables:
            dec = InArgument(datatype, var.expr)
            decs.append(Declare(datatype, dec))

        self.update()

        return decs

class DelStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.variables = kwargs.pop('variables')

        super(DelStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        lines = []
        for var in self.variables:
            if var in namespace:
                namespace.pop(var)
            elif var in stack:
                stack.pop(var)
            else:
                raise Exception('Unknown variable "{}" at position {}'
                                .format(var, self._tx_position))

            line = "del " + str(var)
            lines.append(line)

        self.update()

        return lines

class PassStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.label = kwargs.pop('label')

        super(PassStmt, self).__init__(**kwargs)

    @property
    def expr(self):

        self.update()

        return self.label

class IfStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.body_true  = kwargs.pop('body_true')
        self.body_false = kwargs.pop('body_false')
        self.test       = kwargs.pop('test')

        super(IfStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        ls = []
        for stmt in self.body_true.stmts:
            ls += stmt.local_vars
            ls += stmt.stmt_vars
        if not self.body_false==None:
            for stmt in self.body_false.stmts:
                ls += stmt.local_vars
                ls += stmt.stmt_vars
        return ls

    @property
    def expr(self):
        self.update()
        test       = self.test.expr
        body_true  = self.body_true .expr
        if not self.body_false==None:
            body_false = self.body_false.expr
            return If((test, body_true), (True, body_false))
        else:
            return If((test, body_true))

class AssignStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """

        self.lhs = kwargs.pop('lhs')
        self.rhs = kwargs.pop('rhs')
        self.trailer = kwargs.pop('trailer', None)

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        return [self.lhs]

    def update(self):
        # TODO default type?
        datatype = 'float'
#        datatype = 'int'
        if isinstance(self.rhs, Expression):
            expr = self.rhs
            symbols = set([])
            if isinstance(expr, Basic):
                symbols = expr.free_symbols

            for s in symbols:
                if s.name in namespace:
                    if s.is_integer:
                        datatype = 'int'
                        break
                    elif s.is_Boolean:
                        datatype = 'bool'
                        break

        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)

            # TODO check if var is a return value
            rank = 0
            d_var=Check_type(self.lhs,self.rhs)
            insert_variable(var_name,rank=d_var['rank'],
                            datatype=d_var['datatype'],
                            allocatable=d_var['allocatable'],
                            shape=d_var['shape'])


    @property
    def expr(self):
        if isinstance(self.rhs, Expression):
            rhs = self.rhs.expr

            if isinstance(rhs, Function):
                name = str(type(rhs).__name__)
                F = namespace[name]
                f_expr = F.expr
                results = f_expr.results
                result = results[0]
                insert_variable(self.lhs, \
                                datatype=result.dtype, \
                                allocatable=result.allocatable, \
                                shape=result.shape, \
                                rank=result.rank)
        else:
            rhs = sympify(self.rhs)

        if self.trailer is None:
            l = sympify(self.lhs)
        else:
            args = self.trailer.expr
            l = IndexedVariable(str(self.lhs))[args]

        l = Assign(l, rhs)

        self.update()
        return l

class MultiAssignStmt(BasicStmt):
    """Class representing multiple assignments. In fortran, this correspondans
    to the call of a subroutine"""
    def __init__(self, **kwargs):
        """
        """

        self.lhs     = kwargs.pop('lhs')
        self.name    = kwargs.pop('name')
        self.trailer = kwargs.pop('trailer', None)

        super(MultiAssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        return self.lhs

    def update(self):
        datatype = 'float'
        name = str(self.name)
        if not(name in namespace):
            raise Exception('Undefined function/subroutine {}'.format(name))
        else:
            F = namespace[name]
            if not(isinstance(F, FunctionDefStmt)):
                raise Exception('Expecting a {0} for {1}'.format(type(F), name))

        for var_name in self.lhs:
            if not(var_name in namespace):
                if DEBUG:
                    print("> Found new variable " + var_name)

                # TODO get info from FunctionDefStmt
                rank = 0
                insert_variable(var_name, datatype=datatype, rank=rank)

    @property
    def expr(self):
        self.update()
        lhs = self.lhs
        rhs = self.name
        if not(self.trailer is None):
            args = self.trailer.expr
        else:
            raise Exception('Expecting a trailer')

        return MultiAssign(lhs, rhs, args)


class ForStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.iterable = kwargs.pop('iterable')
        self.start    = kwargs.pop('start')
        self.end      = kwargs.pop('end')
        self.body     = kwargs.pop('body')
        self.step     = kwargs.pop('step', None)

        super(ForStmt, self).__init__(**kwargs)

    @property
    def local_vars(self):
        """."""
        return [self.iterable]

    @property
    def stmt_vars(self):
        """."""
        ls = []
        for stmt in self.body.stmts:
            ls += stmt.local_vars
            ls += stmt.stmt_vars
        return ls

    def update(self):
        # check that start and end were declared, if they are symbols
        insert_variable(self.iterable, datatype='int')

        # TODO do we have to insert them? or just to check they exist?
        insert_variable(self.start,    datatype='int')
        insert_variable(self.end,      datatype='int')
        if not(self.step is None):
            insert_variable(self.step, datatype='int')

    @property
    def expr(self):
        i = Symbol(self.iterable, integer=True)

        if self.start in namespace:
            b = namespace[self.start]
        else:
            try:
                b = Symbol(self.start, integer=True)
            except:
                b = int(self.start)

        if self.end in namespace:
            e = namespace[self.end]
        else:
            try:
                e = Symbol(self.end, integer=True)
            except:
                e = int(self.end)

        if self.step is None:
            s = 1
        else:
            if self.step in namespace:
                s = namespace[self.step]
            else:
                try:
                    s = Symbol(self.step, integer=True)
                except:
                    s = int(self.step)

        self.update()

        body = self.body.expr

        return For(i, (b,e,s), body)

class WhileStmt(BasicStmt):

    def __init__(self, **kwargs):
        """
        """
        self.test     = kwargs.pop('test')
        self.body     = kwargs.pop('body')

        super(WhileStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        ls = []
        for stmt in self.body.stmts:
            ls += stmt.local_vars
            ls += stmt.stmt_vars
        return ls

    @property
    def expr(self):
        test = self.test.expr

        self.update()

        body = self.body.expr

        return While(test, body)

class ExpressionElement(object):
    """Class representing an element of an expression."""
    def __init__(self, **kwargs):

        # textX will pass in parent attribute used for parent-child
        # relationships. We can use it if we want to.
        self.parent = kwargs.get('parent', None)

        # We have 'op' attribute in all grammar rules
        self.op = kwargs['op']


        super(ExpressionElement, self).__init__()

class FactorSigned(ExpressionElement, BasicStmt):
    """Class representing a signed factor."""
    def __init__(self, **kwargs):
        self.sign    = kwargs.pop('sign', '+')
        self.trailer = kwargs.pop('trailer', None)

        super(FactorSigned, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print "> FactorSigned "
        expr = self.op.expr



        if self.trailer is None:
            return -expr if self.sign == '-' else expr
        else:
            args = self.trailer.expr
            if self.trailer.args:
                expr = Function(str(expr))(*args)
            elif self.trailer.subs:
                expr = IndexedVariable(str(expr))[args]
            return -expr if self.sign == '-' else expr

class FactorUnary(ExpressionElement, BasicStmt):
    """Class representing a unary factor."""
    def __init__(self, **kwargs):
        # name of the unary operator

        self.name = kwargs['name']
        self.trailer = kwargs.pop('trailer', None)


        super(FactorUnary, self).__init__(**kwargs)



    @property
    def expr(self):

        if DEBUG:
            print "> FactorUnary "
        expr = self.op.expr
        rhs=expr

        if self.name=='len':
            import ast
            try:
                rhs=ast.literal_eval(expr)
            except:
                rhs=expr
            return LEN(rhs)
        elif self.name=='abs':
            return Abs(rhs)
        elif self.name=='cos':
            return cos(rhs)
        elif self.name=='sin':
            return sin(rhs)
        elif self.name=='sqrt':
            return sqrt(rhs)
        elif self.name=='exp':
            return exp(rhs)
        elif self.name=='log':
            return log(rhs)
        elif self.name=='sign':
            return sign(rhs)
        elif self.name=='csc':
            return csc(rhs)
        elif self.name=='sec':
            return sec(rhs)
        elif self.name=='tan':
            return tan(rhs)
        elif self.name=='cot':
            return cot(rhs)
        elif self.name=='asin':
            return asin(rhs)
        elif self.name=='acsc':
            return acsc(rhs)
        elif self.name=='acos':
            return acos(rhs)
        elif self.name=='asec':
            return asec(rhs)
        elif self.name=='atan':
            return atan(rhs)
        elif self.name=='acot':
            return acot(rhs)
        elif self.name=='atan2':
            return atan2(rhs)
        elif self.name=='factorial':
            return factorial(rhs)
        else:
            raise Exeption('function note supported')


        if self.trailer is None:
            return expr
        else:
            args = self.trailer.expr
            if self.trailer.args:
                expr = Function(str(expr))(*args)
            elif self.trailer.subs:
                expr = IndexedVariable(str(expr))[args]
            return expr

class FactorBinary(ExpressionElement):
    def __init__(self, **kwargs):
        # name of the unary operator
        self.name = kwargs['name']

        super(FactorBinary, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print "> FactorBinary "
#        print self.op

        expr_l = self.op[0].expr
        expr_r = self.op[1].expr

        if self.name == "pow":
            return Pow(expr_l, expr_r)
        elif self.name == "rational":
            return Rational(expr_l, expr_r)
        elif self.name == "dot":
            return Dot(expr_l, expr_r)
        elif self.name == "max":
            return Max(expr_l, expr_r)
        elif self.name == "min":
            return Min(expr_l, expr_r)
        else:
            raise Exception('Unknown variable "{}" at position {}'
                            .format(op, self._tx_position))

class Term(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> Term "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            if operation == '*':
                ret *= operand.expr
            else:
                ret /= operand.expr
        return ret

class Expression(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> Expression "

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):

            if operation == '+':
                ret += operand.expr
            else:
                ret -= operand.expr

        return ret


class Operand(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> Operand "
            print "> stack : ", stack
            print self.op
#        op = self.op[0]
        op = self.op
        if type(op) == float:
            if sympify(op).is_integer:
#                print "> found int ",Integer(op)
                return Integer(op)
            else:
#                print "> found float ",Float(op)
                return Float(op)
        elif type(op) == list:
            # op is a list
            for O in op:
                if O in namespace:
                    if isinstance(namespace[O], Number):
                        return namespace[O].expr
                    else:
                        return namespace[O]
                elif O in stack:
                    if DEBUG:
                        print ">>> found local variables: " + O
                    return Symbol(O)
                elif type(O) == int:
                    return Integer(O)
                elif type(O) == float:
                    return Float(O)
                else:
                    raise Exception('Unknown variable "{}" at position {}'
                                    .format(O, self._tx_position))
        elif isinstance(op, ExpressionElement):
            return op.expr
        elif op in stack:
            if DEBUG:
                print ">>> found local variables: " + op
            return Symbol(op)
        elif op in namespace:
            if isinstance(namespace[op], Number):
                return namespace[op].expr
            if isinstance(namespace[op], FunctionDefStmt):
                return Function(op) #(Symbol(args[0]), Symbol(args[1]))
            else:
                return namespace[op]
        elif(type(op)==unicode):
            return op
        else:
            raise Exception('Undefined variable "{}"'.format(op))

class Test(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op.expr
        return ret

# TODO improve using sympy And, Or, Not, ...
class OrTest(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            ret = (ret or operand.expr)
        return ret

# TODO improve using sympy And, Or, Not, ...
class AndTest(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            ret = (ret and operand.expr)
        return ret

# TODO improve using sympy And, Or, Not, ...
class NotTest(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op.expr
        ret = (not ret)
        return ret

class Comparison(ExpressionElement):
    # TODO ARA finish
    @property
    def expr(self):
        if DEBUG:
            print "> Comparison "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
#            print "Comparison : ", ret, operation, operand.expr
            if operation == "==":
                ret = EqualityStmt(ret, operand.expr)
            elif operation == ">":
                ret = Gt(ret, operand.expr)
            elif operation == ">=":
                ret = Ge(ret, operand.expr)
            elif operation == "<":
                ret = Lt(ret, operand.expr)
            elif operation == "<=":
                ret = Le(ret, operand.expr)
            elif operation == "<>":
                ret = NotequalStmt(ret, operand.expr)
            else:
                raise Exception('operation not yet available at position {}'
                                .format(self._tx_position))
        return ret

class FlowStmt(BasicStmt):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        self.label = kwargs.pop('label')

class BreakStmt(FlowStmt):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(BreakStmt, self).__init__(**kwargs)

class ContinueStmt(FlowStmt):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(ContinueStmt, self).__init__(**kwargs)

class ReturnStmt(FlowStmt):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        self.variables = kwargs.pop('variables')
        self.results   = None

        super(ReturnStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        """
        self.update()

        decs = []
        # TODO depending on additional options from the grammar
        # TODO check that var is in namespace
        for var_name in self.variables:
            if var_name in variables:
                var = variables[var_name]
                if isinstance(var, Variable):
                    res = Result(var.dtype, var_name, \
                           rank=var.rank, \
                           allocatable=var.allocatable, \
                           shape=var.shape)
                else:
                    datatype = var.datatype
                    res = Result(datatype, var_name)
            else:
                datatype = 'float'
                res = Result(datatype, var_name)
                raise()

            decs.append(res)

        self.results = decs
        return decs

class RaiseStmt(FlowStmt):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(RaiseStmt, self).__init__(**kwargs)

class YieldStmt(FlowStmt):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        super(YieldStmt, self).__init__(**kwargs)

class FunctionDefStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')
        self.body = kwargs.pop('body')

        # TODO improve
        namespace[str(self.name)] = self

        super(FunctionDefStmt, self).__init__(**kwargs)

    def update(self):
        for arg_name in self.args:
            if not(arg_name in namespace):
                if DEBUG:
                    print("> Found new argument" + arg_name)

                # TODO define datatype, rank
                # TODO check if arg is a return value
                rank = 0
                datatype = 'float'
                insert_variable(arg_name, datatype=datatype, rank=rank,
                                is_argument=True)
            else:
                print("+++ found already declared argument : ", arg_name)

    @property
    def local_vars(self):
        """."""
        return self.body.local_vars

    @property
    def stmt_vars(self):
        """."""
        return self.body.stmt_vars

    @property
    def expr(self):
        self.update()
        body = self.body.expr

        name = str(self.name)

        args    = [variables[arg_name] for arg_name in self.args]
        prelude = [declarations[arg_name] for arg_name in self.args]

        # ...
        results = []
        for stmt in self.body.stmts:
            if isinstance(stmt, ReturnStmt):
                results += stmt.results
        # ...

        # ... cleaning the namespace
        for arg_name in self.args:
            declarations.pop(arg_name)
            variables.pop(arg_name)
            namespace.pop(arg_name)

        ls = self.local_vars + self.stmt_vars
        for var_name in ls:
            if var_name in namespace:
                namespace.pop(var_name, None)
                variables.pop(var_name, None)
                dec = declarations.pop(var_name, None)
                if dec:
                    prelude.append(dec)
        # ...

        body = prelude + body

        local_vars  = []
        global_vars = []

        return FunctionDef(name, args, results, body, local_vars, global_vars)



class NumpyZerosStmt(AssignStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """

        self.lhs        = kwargs.pop('lhs')
        self.parameters = kwargs.pop('parameters')
       # print(self.parameters[0].value,'####')
        #raise SystemExit()

        labels = [str(p.label) for p in self.parameters]
#        values = [p.value.value for p in self.parameters]
        values = []
        for p in self.parameters:
            try:
                v = p.value.value.args
            except:
                v = p.value.value
            values.append(v)
        d = {}
        for (label, value) in zip(labels, values):
            d[label] = value
        self.parameters = d

        try:
            self.datatype = self.parameters['dtype']
        except:
            self.datatype = 'float'

        try:
            self.shape = self.parameters['shape']
        except:
            raise Exception('Expecting shape at position {}'
                            .format(self._tx_position))

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        return [self.lhs]

    def update(self):
        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)

            datatype = self.datatype

            rank = 0
            if isinstance(self.shape, int):
                shape = self.shape
                rank = 1
            elif isinstance(self.shape, float):
                shape = int(self.shape)
                rank = 1
            elif isinstance(self.shape, list):
                shape = []
                for s in self.shape:
                    if isinstance(s, (int, float)):
                        shape.append(int(s))
                    elif isinstance(s, str):
                        if not(s in namespace):
                            raise Exception('Could not find shape variable.')

#                        if not(variables[s].dtype == 'int'):
#                            raise Exception('Shape must be an integer.')

                        shape.append(namespace[s])
                    elif isinstance(s,FactorUnary):
                        shape.append(s.expr)


                    else:
                        raise TypeError('Expecting a int, float or string')
                rank = len(shape)
            elif isinstance(self.shape,FactorUnary):
                 shape=self.shape.expr
            else:
                shape = str(self.shape)
                if shape in namespace:
                    shape = namespace[shape]
                    # TODO compute rank
                    rank = 1
                else:
                    raise Exception('Wrong instance for shape.')
            self.shape = shape

            if datatype is None:
                if DEBUG:
                    print("> No Datatype is specified, int will be used.")
                datatype = 'int'
            # TODO check if var is a return value
            insert_variable(var_name, \
                            datatype=datatype, \
                            rank=rank, \
                            allocatable=True,shape = self.shape)

    @property
    def expr(self):
        self.update()

        shape = self.shape

        var_name = self.lhs
        var = Symbol(var_name)

        stmt = NumpyZeros(var, shape)

        return stmt

class NumpyZerosLikeStmt(AssignStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs = kwargs.pop('lhs')
        self.rhs = kwargs.pop('rhs')

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        return [self.lhs]

    def update(self):
        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)
        v=variables[self.rhs]


        insert_variable(var_name, \
                            datatype=v.dtype, \
                            rank=v.rank, \
                            allocatable=v.allocatable,shape=v.shape)





    @property
    def expr(self):
        self.update()
        v=variables[self.rhs]
        shape = v.shape



        if shape==None:
            shape=1

        var_name = self.lhs
        var = Symbol(var_name)

        stmt = NumpyZeros(var, shape)

        return stmt

class NumpyOnesStmt(AssignStmt):

    def __init__(self, **kwargs):
        """
        """
        self.lhs        = kwargs.pop('lhs')
        self.parameters = kwargs.pop('parameters')

        labels = [str(p.label) for p in self.parameters]
#        values = [p.value.value for p in self.parameters]
        values = []
        for p in self.parameters:
            try:
                v = p.value.value.args
            except:
                v = p.value.value
            values.append(v)
        d = {}
        for (label, value) in zip(labels, values):
            d[label] = value
        self.parameters = d

        try:
            self.datatype = self.parameters['dtype']
        except:
            self.datatype = 'float'

        try:
            self.shape = self.parameters['shape']
        except:
            raise Exception('Expecting shape at position {}'
                            .format(self._tx_position))

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        return [self.lhs]

    def update(self):
        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)

            datatype = self.datatype

            rank = 0
            if isinstance(self.shape, int):
                shape = self.shape
                rank = 1
            elif isinstance(self.shape, float):
                shape = int(self.shape)
                rank = 1
            elif isinstance(self.shape, list):
                shape = [int(s) for s in self.shape]
                rank = len(shape)
            else:
                raise Exception('Wrong instance for shape.')
            self.shape = shape

            if datatype is None:
                if DEBUG:
                    print("> No Datatype is specified, int will be used.")
                datatype = 'int'
            # TODO check if var is a return value
            insert_variable(var_name, \
                            datatype=datatype, \
                            rank=rank, \
                            allocatable=True,shape=self.shape)

    @property
    def expr(self):
        self.update()

        shape = self.shape

        var_name = self.lhs
        var = Symbol(var_name)

        stmt = NumpyOnes(var, shape)

        return stmt

class NumpyLinspaceStmt(AssignStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs   = kwargs.pop('lhs')
        self.start = kwargs.pop('start')
        self.end   = kwargs.pop('end')
        self.size  = kwargs.pop('size')

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """."""
        return [self.lhs]

    def update(self):
        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)

            s    = self.start
            e    = self.end
            size = self.size

            ls = [s, e, size]
            for name in ls:
                if isinstance(name, (int, float)):
                    pass
                elif not(name in namespace):
                    raise Exception('Unknown variable "{}" at position {}'
                                    .format(name, self._tx_position))

            var = Symbol(var_name)

            namespace[var_name] = var

            # TODO improve
            datatype = 'float'

            dec = Variable(datatype, var, rank=1)
            self.declarations.append(Declare(datatype, dec))

    @property
    def expr(self):
        self.update()

        var_name = self.lhs
        var = Symbol(var_name)

        start = self.start
        end   = self.end
        size  = self.size

        stmt = NumpyLinspace(var, start, end, size)

        return stmt

class NumpyArrayStmt(AssignStmt):
    def __init__(self, **kwargs):


        self.lhs= kwargs.pop('lhs')
        self.rhs= kwargs.pop('rhs')
        import ast
        self.rhs=ast.literal_eval(self.rhs)
        self.dtype=kwargs.pop('dtype')

        import numpy as n
        self.shape=n.shape(self.rhs)
        super(AssignStmt, self).__init__(**kwargs)


    @property
    def stmt_vars(self):
        """."""
        return [self.lhs]


    @property
    def expr(self):
        self.update()
        var=sympify(self.lhs)
        mylist=self.rhs

        if isinstance(mylist[0],list):
            if self.dtype=='int':
                mylist=[map(int, i) for i in mylist]
            elif self.dtype=='float':
                mylist=[map(float, i) for i in mylist]
        else:
            if self.dtype=='int':
                mylist=map(int,mylist)
            elif self.dtype=='float':
                mylist=map(float,mylist)


        return NumpyArray(var,mylist,self.shape)
    def update(self):
        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)

        rank=len(self.shape)

        datatype=str(self.dtype)
        if self.dtype is None:
            datatype='float'
            #TODO improve later
        var=Symbol(var_name)
        insert_variable(var_name, \
                            datatype=datatype, \
                            rank=rank, \
                            allocatable=True,shape=self.shape)


class ImportFromStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.dotted_name     = kwargs.pop('dotted_name')
        self.import_as_names = kwargs.pop('import_as_names')

        super(ImportFromStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()

        # TODO how to handle dotted packages?
        fil = self.dotted_name.names[0]
        funcs = self.import_as_names.names
        return Import(fil, funcs)

class PythonPrintStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')

        super(PythonPrintStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()

        func_name   = self.name
        args        = self.args
        expressions=[]

        for arg in args:
            if not isinstance(arg,str):
               expressions.append(arg.expr)
            else:
                expressions.append(arg)
        return Print(expressions)

class CommentStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.text = kwargs.pop('text')

        # TODO improve
        #      to remove:  # coding: utf-8
        if ("coding:" in self.text) or ("utf-8" in self.text):
            self.text = ""

        super(CommentStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()
        return Comment(self.text)

class SuiteStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.stmts = kwargs.pop('stmts')

        super(SuiteStmt, self).__init__(**kwargs)

    @property
    def local_vars(self):
        """."""
        ls = []
        for stmt in self.stmts:
            ls += stmt.local_vars
        s = set(ls)
        return list(s)

    @property
    def stmt_vars(self):
        """."""
        ls = []
        for stmt in self.stmts:
            ls += stmt.stmt_vars
        s = set(ls)
        return list(s)

    def update(self):
        pass

    @property
    def expr(self):
        self.update()
        ls = [stmt.expr for stmt in  self.stmts]
        return ls

class BasicTrailer(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.args = kwargs.pop('args', None)

        super(BasicTrailer, self).__init__(**kwargs)

    @property
    def expr(self):
        pass

class Trailer(BasicTrailer):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.subs = kwargs.pop('subs', None)

        super(Trailer, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()
        if self.args:
            return self.args.expr
        if self.subs:
            return self.subs.expr

class TrailerArgList(BasicTrailer):
    """Class representing arguments of a function call."""
    def __init__(self, **kwargs):
        """
        """
        super(TrailerArgList, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()
        return [arg.expr for arg in  self.args]

class TrailerSubscriptList(BasicTrailer):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        super(TrailerSubscriptList, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()
        args = []
        for a in self.args:
            if isinstance(a, Expression):
                arg = do_arg(a)

                # TODO treat n correctly
                n = Symbol('n', integer=True)
                i = Idx(arg, n)
                args.append(i)
            elif isinstance(a, BasicSlice):
                arg = a.expr
                args.append(arg)
            else:
                raise Exception('Wrong instance')
        return args

class BasicSlice(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.start = kwargs.pop('start', None)
        self.end   = kwargs.pop('end',   None)

        super(BasicSlice, self).__init__(**kwargs)

    def extract_arg(self, name):
        if name is None:
            return None

        name = name.expr
        var = None
        if isinstance(name, (Integer, Float)):
            var = Integer(name)
        else:
            if name in namespace:
                var = namespace[name]
            else:
                print("stop here 2")
                print type(name), name
                import sys; sys.exit(0)

        return var

    @property
    def expr(self):
        start = self.extract_arg(self.start)
        end   = self.extract_arg(self.end)

        return Slice(start, end)

class TrailerSlice(BasicSlice):
    """Class representing a ."""
    pass

class TrailerSliceRight(BasicSlice):
    """Class representing a ."""
    pass

class TrailerSliceLeft(BasicSlice):
    """Class representing a ."""
    pass

class ThreadStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs  = kwargs.pop('lhs')
        self.func = kwargs.pop('func')

        super(ThreadStmt, self).__init__(**kwargs)

    def update(self):
        var_name = str(self.lhs)
        if not(var_name in namespace):
            insert_variable(var_name, datatype='int', rank=0)
        else:
            raise Exception('Already declared variable for thread_id.')

    @property
    def expr(self):
        self.update()

        var_name = str(self.lhs)
        var = Symbol(var_name)

        func = str(self.func)
        if func == 'thread_id':
            return ThreadID(var)
        elif func == 'thread_number':
            return ThreadsNumber(var)
        else:
            raise Exception('Wrong value for func.')
