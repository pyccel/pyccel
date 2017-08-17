# coding: utf-8

from sympy import Symbol, sympify
from symcc.types.ast import For, Assign


DEBUG = False
#DEBUG = True

__all__ = ["Pyccel", \
           "Expression", "Term", "Operand", \
           "FactorSigned", "FactorUnary", "FactorBinary", \
           # statements
           "AssignStmt", "ForStmt", "DeclarationStmt", \
           "DelStmt", "PassStmt" \
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

class DeclarationStmt(object):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        A Real number is defined by

        * name

        .. note::
            The grammar rule to define a Real is

            Real:
            "Real" DEF name=ID
            ;
        """
        self.variables = kwargs.pop('variables')
        self.datatype = kwargs.pop('datatype')

    @property
    def expr(self):
        ls = []
        for v in self.variables:
            ls.append(Symbol(var))
        return ls

class DelStmt(object):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        A Real number is defined by

        * name

        .. note::
            The grammar rule to define a Real is

            Real:
            "Real" DEF name=ID
            ;
        """
        self.variables = kwargs.pop('variables')

    @property
    def expr(self):
        lines = []
        for var in self.variables:
            line = "del " + str(var)
            lines.append(line)
        return lines

# TODO not working
class PassStmt(object):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        A Real number is defined by

        * name

        .. note::
            The grammar rule to define a Real is

            Real:
            "Real" DEF name=ID
            ;
        """
        self.name_stmt = kwargs.pop('name')

    @property
    def expr(self):
        return self.name_stmt

class AssignStmt(object):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.lhs = kwargs.pop('lhs')
        self.rhs = kwargs.pop('rhs')

#        namespace[self.iterable.name] = self.iterable

    @property
    def expr(self):
        if isinstance(self.rhs, Expression):
            rhs = sympify(self.rhs.expr)
        else:
            rhs = sympify(self.rhs)

        lhs = sympify(self.lhs)

        return Assign(lhs, rhs)

class ForStmt(object):
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

#        namespace[self.iterable.name] = self.iterable

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

        # TODO body must be parsed as a list
        body = [self.body.expr]
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
                    # TODO use isinstance
                    if type(namespace[O]) in [Real]:
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
            # TODO use isinstance
            if type(namespace[op]) in [Real]:
                return namespace[op].expr
            else:
                return namespace[op]
        else:
            raise Exception('Unknown variable "{}" at position {}'
                            .format(op, self._tx_position))

