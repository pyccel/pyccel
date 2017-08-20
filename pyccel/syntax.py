# coding: utf-8

from sympy import Symbol, sympify, Piecewise

from symcc.types.ast import For, Assign, Declare, Variable
from symcc.types.ast import Argument, InArgument, InOutArgument, Result
from symcc.types.ast import FunctionDef
from symcc.types.ast import Import
from symcc.types.ast import NumpyZeros

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
           "NumpyOnesStmt", "NumpyLinspaceStmt"
           ]


# Global variable namespace
namespace = {}
stack     = {}
settings  = {}

operators = {}

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
        self.statements = []

    def update(self):
        pass

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

        super(AssignStmt, self).__init__(**kwargs)

    def update(self):
        for var_name in self.lhs:
            if not(var_name in namespace):
                if DEBUG:
                    print("> Found new variable " + var_name)

                var = Symbol(var_name)
                namespace[var_name] = var
                datatype = 'int'
                # TODO define datatype
                # TODO check if var is a return value
                dec = Variable(datatype, var)
                self.statements.append(Declare(datatype, dec))

    @property
    def expr(self):
        if isinstance(self.rhs, Expression):
            rhs = sympify(self.rhs.expr)
        else:
            rhs = sympify(self.rhs)

        ls = []
        for l in self.lhs:
            ls.append(sympify(l))

        ls = [Assign(l, rhs) for l in ls]

        self.update()

        if len(ls) == 1:
            return ls[0]
        else:
            return ls


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

        namespace[self.iterable] = Symbol(self.iterable)

        super(ForStmt, self).__init__(**kwargs)

    def update(self):
        i   = Symbol(self.iterable, integer=True)
        dec = Variable('int', i)
        self.statements.append(Declare('int', dec))

        body = []
        for stmt in self.body:
            if isinstance(stmt, list):
                body += stmt
            else:
                body.append(stmt)

        for stmt in body:
            e = stmt.expr
            self.statements += stmt.statements

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


class FactorSigned(ExpressionElement):
    """Class representing a signed factor."""
    def __init__(self, **kwargs):
        self.sign = kwargs.pop('sign', '+')
        super(FactorSigned, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print "> FactorSigned "
        expr = self.op.expr
        return -expr if self.sign == '-' else expr

class FactorUnary(ExpressionElement):
    """Class representing a unary factor."""
    def __init__(self, **kwargs):
        # name of the unary operator
        self.name = kwargs['name']

        super(FactorUnary, self).__init__(**kwargs)

    @property
    def expr(self):
        if DEBUG:
            print "> FactorUnary "
        expr = self.op.expr
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
                ret *= sympify(operand.expr)
            else:
                ret /= sympify(operand.expr)
        return ret


class Expression(ExpressionElement):
    @property
    def expr(self):
        if DEBUG:
            print "> Expression "
        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            if operation == '+':
                ret += sympify(operand.expr)
            else:
                ret -= sympify(operand.expr)
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
        if type(op) in {int, float}:
            return op
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
                self.statements.append(Declare(datatype, dec))

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
        prelude = self.statements
        for stmt in self.body:
            if not(isinstance(stmt, ReturnStmt)):
                prelude += stmt.statements
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
        values = [p.value for p in self.parameters]
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
            # TODO ARA fix in symcc
            self.shape = int(self.shape)
        except:
            raise Exception('Expecting shape at position {}'
                            .format(self._tx_position))

        super(AssignStmt, self).__init__(**kwargs)

    def update(self):
        for var_name in self.lhs:
            if not(var_name in namespace):
                if DEBUG:
                    print("> Found new variable " + var_name)

                datatype = self.datatype
                shape    = self.shape

                var = Symbol(var_name)
                namespace[var_name] = var
                if datatype is None:
                    if DEBUG:
                        print("> No Datatype is specified, int will be used.")
                    datatype = 'int'
                # TODO check if var is a return value

                rank = 0
                if type(shape) == int:
                    rank = 1
                else:
                    raise Exception('Only rank=1 is available')

                dec = Variable(datatype, var, rank=rank)
                self.statements.append(Declare(datatype, dec))

    @property
    def expr(self):
        self.update()

        shape = self.shape

        stmts = []
        for var_name in self.lhs:
            var = Symbol(var_name)

            stmt = NumpyZeros(var, shape)
            stmts.append(stmt)

        return stmts

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
