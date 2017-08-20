# coding: utf-8

from sympy import Symbol, sympify, Piecewise, Integer, Float, Add, Mul
from sympy import true, false
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.core.basic import Basic
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge

from pyccel.types.ast import For, Assign, Declare, Variable
from pyccel.types.ast import Argument, InArgument, InOutArgument, Result
from pyccel.types.ast import FunctionDef
from pyccel.types.ast import Import
from pyccel.types.ast import NumpyZeros

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
           # numpy statments
           "NumpyZerosStmt", "NumpyZerosLikeStmt", \
           "NumpyOnesStmt", "NumpyLinspaceStmt", \
           # Test
           "Test", "OrTest", "AndTest", "NotTest", "Comparison"
           ]


# Global variable namespace
namespace = {}
stack     = {}
settings  = {}

operators = {}

namespace["True"]  = true
namespace["False"] = false

class Pyccel(object):
    """Class for Pyccel syntax."""
    def __init__(self, **kwargs):
        """
        Constructor for Pyccel.

        """
        try:
            self.declarations = kwargs.pop('declarations')
        except:
            self.declarations = []
        try:
            self.statements = kwargs.pop('statements')
        except:
            self.statements = []

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
        self.declarations = []
        self.statements   = []

    def update(self):
        pass

    # TODO move somewhere else
    def do_trailer(self, trailer):
        args = []
        for a in trailer.args:
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
                    rhs = a.expr
                    # TODO ARA
                    name = 'result_%d' % abs(hash(rhs))
                    arg = Symbol(name, integer=True)
                    var = Variable('int', arg)
                    self.declarations.append(Declare('int', var))
                    self.statements.append(Assign(arg, rhs))

#                if not(isinstance(arg, Symbol)):
#                    print "-----"
#                    print type(arg), arg
#                    arg = Integer(arg)
#            elif isinstance(a, Basic):
#                arg = a
            else:
                arg = Integer(a)

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

        # TODO allow list of stmts
        print "======="
        print test
        print ls
        print rs
        e = Piecewise((ls[0], test), (rs[0], True))
        print "> IfStmt: TODO handle a list of statments"
#        e = Piecewise((ls, test), (rs, True))

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

            var = Symbol(var_name)
            namespace[var_name] = var
            # TODO check if var is a return value
            rank = 0
            dec = Variable(datatype, var, rank=rank)
            self.declarations.append(Declare(datatype, dec))

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
            l = IndexedBase(str(self.lhs))[args]

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

        namespace[self.iterable] = Symbol(self.iterable, integer=True)

        super(ForStmt, self).__init__(**kwargs)

    def update(self):
        i   = Symbol(self.iterable, integer=True)
        dec = Variable('int', i)
        self.declarations.append(Declare('int', dec))

        body = []
        for stmt in self.body:
            if isinstance(stmt, list):
                body += stmt
            else:
                body.append(stmt)

        for stmt in body:
            e = stmt.expr
            self.declarations += stmt.declarations

    @property
    def expr(self):
        i = Symbol(self.iterable, integer=True)

        try:
            b = Symbol(self.start, integer=True)
        except:
            b = int(self.start)

        try:
            e = Symbol(self.end, integer=True)
        except:
            e = int(self.end)

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
            expr = IndexedBase(str(expr))[args]
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
            expr = IndexedBase(str(expr))[args]
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

        if self.name == "dot":
            return dot(expr_l, expr_r)
        elif self.name == "inner":
            return inner(expr_l, expr_r)
        elif self.name == "outer":
            return outer(expr_l, expr_r)
        elif self.name == "cross":
            return cross(expr_l, expr_r)
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
    # TODO ARA implement
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
            elif operation == "not":
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
            self.datatype = 'int'

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

            var = Symbol(var_name)

            namespace[var_name] = var
            if datatype is None:
                if DEBUG:
                    print("> No Datatype is specified, int will be used.")
                datatype = 'int'
            # TODO check if var is a return value

            dec = Variable(datatype, var, rank=rank)
            self.declarations.append(Declare(datatype, dec))

    @property
    def expr(self):
        self.update()

        shape = self.shape

        var_name = self.lhs
        var = Symbol(var_name)
#        var = IndexedBase(var_name)

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

    @property
    def expr(self):
        self.update()
        return ""

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
