# coding: utf-8

from sympy import Symbol, sympify, Piecewise, Integer, Float, Add, Mul
from sympy import true, false
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.core.basic import Basic
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.power import Pow

from pyccel.types.ast import For, Assign, Declare, Variable
from pyccel.types.ast import Argument, InArgument, InOutArgument, Result
from pyccel.types.ast import FunctionDef
from pyccel.types.ast import Import
from pyccel.types.ast import Print
from pyccel.types.ast import Comment
from pyccel.types.ast import AnnotatedComment
from pyccel.types.ast import IndexedVariable
from pyccel.types.ast import Slice
from pyccel.types.ast import NumpyZeros, NumpyLinspace

DEBUG = False
#DEBUG = True

__all__ = ["Pyccel", \
           "Expression", "Term", "Operand", \
           "FactorSigned", "FactorUnary", "FactorBinary", \
           # statements
           "AssignStmt", "DeclarationStmt", \
           # compound stmts
           "ForStmt", "IfStmt", \
           # Flow statements
           "FlowStmt", "BreakStmt", "ContinueStmt", \
           "RaiseStmt", "YieldStmt", "ReturnStmt", \
           "DelStmt", "PassStmt", "FunctionDefStmt", \
           "ImportFromStmt", \
           "CommentStmt", "AnnotatedStmt", \
           # python standard library statements
           "PythonPrintStmt", \
           # numpy statments
           "NumpyZerosStmt", "NumpyZerosLikeStmt", \
           "NumpyOnesStmt", "NumpyLinspaceStmt", \
           # Test
           "Test", "OrTest", "AndTest", "NotTest", "Comparison"
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

def insert_variable(var_name, var=None, datatype=None, rank=0, allocatable=False):
    if type(var_name) in [int, float]:
        return

    if datatype is None:
#        datatype = 'int'
        datatype = 'float'

    is_integer = (datatype == 'int')

    # we first create a sympy symbol
    s = Symbol(var_name, integer=is_integer)

    # we create a variable (for annotation)
    if var is None:
        var = Variable(datatype, s, rank=rank, allocatable=allocatable)

    # we create a declaration for code generation
    dec = Declare(datatype, var)

    if var_name in namespace:
        var_old = variables[var_name]
        if not (var == var_old):
            if True:
            #if DEBUG:
                print ">>> wrong declaration : ", var_name
                print "    type will be changed."

            namespace.pop(var_name)
            variables.pop(var_name)
            declarations.pop(var_name)

            namespace[var_name]    = s
            variables[var_name]    = var
            declarations[var_name] = dec
    else:
        namespace[var_name]    = s
        variables[var_name]    = var
        declarations[var_name] = dec


class Pyccel(object):
    """Class for Pyccel syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Pyccel.

        """
        self.statements   = kwargs.pop('statements',   [])

    @property
    def declarations(self):
        return declarations

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
        self.stmt_vars    = []
        self.local_vars   = []

    @property
    def declarations(self):
        return [declarations[v] for v in self.stmt_vars + self.local_vars]

    @property
    def local_declarations(self):
        return [declarations[v] for v in self.local_vars]

    def update(self):
        pass

    # TODO move somewhere else
    def do_trailer(self, trailer):
        # ...
        def do_arg(a):
            if isinstance(a, str):
                arg = Symbol(a, integer=True)
            elif isinstance(a, Expression):
                arg = a.expr
                try:
                    if not(isinstance(arg, Symbol)):
                        arg = Integer(arg)
                    else:
                        arg = Symbol(arg.name, integer=True)
                except:
                    raise Exception('not available yet')
#                    rhs = a.expr
#                    # TODO ARA
#                    name = 'result_%d' % abs(hash(rhs))
#                    arg = Symbol(name, integer=True)
#                    var = Variable('int', arg)
#                    self.declarations.append(Declare('int', var))
#                    self.statements.append(Assign(arg, rhs))
            else:
                arg = Integer(a)
            return arg
        # ...

        # there are two kind of trailers
        # 1. a symbol, an expression
        # 2. slices
        is_subscript = False
        if trailer.args:
            inputs = trailer.args
        elif trailer.subs:
            inputs = trailer.subs
            is_subscript = True
        else:
            raise Exception('Wrong inputs for the trailer at position {}'
                            .format(var, self._tx_position))

        # only slices of the form a:b are possible
        # this assumes that inputs.args is of length 2
        if is_subscript:
            assert(len(inputs.args) == 2)

            start = do_arg(inputs.args[0])
            end   = do_arg(inputs.args[1])

            args = Slice(start, end)
        else:
            args = []
            for a in inputs.args:
                arg = do_arg(a)

                # TODO treat n correctly
                n = Symbol('n', integer=True)
                i = Idx(arg, n)
                args.append(i)
        return args

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
    def expr(self):
        test = self.test.expr
        ls = [l.expr for l in self.body_true]
        rs = [l.expr for l in self.body_false]

        e = Piecewise((ls, test), (rs, True))

        self.update()

        return e

class AssignStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs = kwargs.pop('lhs')
        self.rhs = kwargs.pop('rhs')
        self.trailer = kwargs.pop('trailer', None)

        super(AssignStmt, self).__init__(**kwargs)

    def update(self):
        datatype = 'float'
        if isinstance(self.rhs, Expression):
            expr = self.rhs.expr
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
            insert_variable(var_name, datatype=datatype, rank=rank)
            self.stmt_vars.append(var_name)

    @property
    def expr(self):
        if isinstance(self.rhs, Expression):
            rhs = sympify(self.rhs.expr)
        else:
            rhs = sympify(self.rhs)

        if self.trailer is None:
            l = sympify(self.lhs)
        else:
            args = self.do_trailer(self.trailer)
            l = IndexedVariable(str(self.lhs))[args]

        l = Assign(l, rhs)

        self.update()
        return l


class ForStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.iterable = kwargs.pop('iterable')
        self.start    = kwargs.pop('start')
        self.end      = kwargs.pop('end')
        self.body     = kwargs.pop('body')

        # TODO add step
        self.step     = 1

        super(ForStmt, self).__init__(**kwargs)

    def update(self):
        # check that start and end were declared, if they are symbols

        insert_variable(self.start,    datatype='int')
        insert_variable(self.end,      datatype='int')
        insert_variable(self.iterable, datatype='int')

        self.local_vars.append(self.iterable)

        # TODO to keep or remove?
        if not(type(self.start) in [int, float]):
            self.local_vars.append(self.start)
        if not(type(self.end) in [int, float]):
            self.local_vars.append(self.end)

#        if not ri:
#            self.declarations.append(declarations[self.iterable])
#
#        if re:
#            dec_new = declarations[self.end]
#            names = [str(v.name) for v in dec_new.variables]
#            if self.end in names:
#                i = 0
#                for dec in self.declarations:
#                    name = str(dec.variables[0].name)
#                    if name == self.end:
#                        break
#                    i +=1
#                    print name, self.end, i
#                if i < len(self.declarations) :
#                    self.declarations[i] = declarations[self.end]
#        else:
#            self.declarations.append(declarations[self.end])

        body = []
        for stmt in self.body:
            if isinstance(stmt, list):
                body += stmt
            else:
                body.append(stmt)

        for stmt in body:
            e = stmt.expr
            # TODO to improve
            self.local_vars += stmt.local_vars
            self.stmt_vars  += stmt.stmt_vars

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

        if self.step in namespace:
            s = namespace[self.step]
        else:
            try:
                s = Symbol(self.step, integer=True)
            except:
                s = int(self.step)

        body = []
        for stmt in self.body:
            if isinstance(stmt, list):
                body += stmt
            else:
                body.append(stmt)

        self.update()

        body = [stmt.expr for stmt in body]
        return For(i, (b,e,s), body)

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
            args = self.do_trailer(self.trailer)
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
        if self.trailer is None:
            return expr
        else:
            args = self.do_trailer(self.trailer)
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
            if (op).is_integer():
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
            else:
                return namespace[op]
        else:
            raise Exception('Unknown variable "{}" at position {}'
                            .format(op, self._tx_position))


class Test(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op.expr
        return ret

class OrTest(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            ret = (ret or operand.expr)
        return ret

class AndTest(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> DEBUG "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            ret = (ret and operand.expr)
        return ret

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
                ret = Eq(ret, operand.expr)
            elif operation == ">":
                ret = Gt(ret, operand.expr)
            elif operation == ">=":
                ret = Ge(ret, operand.expr)
            elif operation == "<":
                ret = Lt(ret, operand.expr)
            elif operation == "<=":
                ret = Le(ret, operand.expr)
            elif operation == "<>":
                ret = Ne(ret, operand.expr)
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
        print "ReturnStmt : ", self.variables

        super(ReturnStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        """
        datatype = 'int'
        decs = []
        # TODO depending on additional options from the grammar
        # TODO check that var is in namespace
        for var in self.variables:
            decs.append(Result(datatype, var))

        self.update()

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

        super(FunctionDefStmt, self).__init__(**kwargs)

    def update(self):
        for arg_name in self.args:
            if not(arg_name in namespace):
                if DEBUG:
                    print("> Found new argument" + arg_name)

                arg = Symbol(arg_name)
                namespace[arg_name] = arg
                datatype = 'int'
                # TODO define datatype
                # TODO check if arg is a return value
                dec = InArgument(datatype, arg)
                self.declarations.append(Declare(datatype, dec))

    @property
    def expr(self):
        name = str(self.name)

        # TODO datatype
        datatype = 'int'

        args = [InArgument(datatype, v) for v in self.args]

        body = []
        for stmt in self.body:
            print type(stmt)
            if isinstance(stmt, list):
                body += stmt
            elif not(isinstance(stmt, ReturnStmt)):
                body.append(stmt)

        self.update()

        body = [stmt.expr for stmt in body]

        results = []
        prelude = self.declarations
        for stmt in self.body:
            if not(isinstance(stmt, ReturnStmt)):
                prelude += stmt.declarations
            else:
                results += stmt.expr
        body = prelude + body

        for arg_name in self.args:
            if (arg_name in namespace):
                namespace.pop(arg_name)

        return FunctionDef(name, args, body, results)

class NumpyZerosStmt(AssignStmt):
    """Class representing a ."""
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
                            allocatable=True)
            self.stmt_vars.append(var_name)

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

    def update(self):
        pass

    @property
    def expr(self):
        self.update()
        return ""

class NumpyOnesStmt(AssignStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs = kwargs.pop('lhs')
        self.shape = kwargs.pop('shape')

        super(AssignStmt, self).__init__(**kwargs)

    def update(self):
        pass

    @property
    def expr(self):
        self.update()
        return ""

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
        expressions = [arg.expr for arg in args]
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

class AnnotatedStmt(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.accel      = kwargs.pop('accel')
        self.do         = kwargs.pop('do', None)
        self.end        = kwargs.pop('end', None)
        self.parallel   = kwargs.pop('parallel', None)
        self.section    = kwargs.pop('section',  None)
        self.visibility = kwargs.pop('visibility', None)
        self.variables  = kwargs.pop('variables',  None)

        super(AnnotatedStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        self.update()

        return AnnotatedComment(accel=self.accel, \
                                do=self.do, \
                                end=self.end, \
                                parallel=self.parallel, \
                                section=self.section, \
                                visibility=self.visibility, \
                                variables=self.variables)
