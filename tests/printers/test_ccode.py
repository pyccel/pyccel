# pylint: disable=missing-function-docstring, missing-module-docstring/
from sympy.core import (pi, oo, symbols, Rational, Integer, Float, GoldenRatio,
        EulerGamma, Catalan, Lambda)
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt, gamma
from sympy.sets.fancysets import Range
from sympy.utilities.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx

from pyccel.types.ast import (Assign, AugAssign, For, InArgument, Result,
        FunctionDef, Return, Import, Declare, Variable)
from pyccel.printers import ccode

x, y, z = symbols('x, y, z')
a, b, c = symbols('a, b, c')


def test_printmethod():
    class fabs(Abs):
        def _ccode(self, printer):
            return "fabs(%s)" % printer._print(self.args[0])
    assert ccode(fabs(x)) == "fabs(x)"


def test_ccode_sqrt():
    assert ccode(sqrt(x)) == "sqrt(x)"
    assert ccode(x**0.5) == "sqrt(x)"
    assert ccode(sqrt(Float(10))) == "3.16227766016838"


def test_ccode_Pow():
    assert ccode(x**3) == "pow(x, 3)"
    assert ccode(x**(y**3)) == "pow(x, pow(y, 3))"
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2) + y)"
    assert ccode(x**-1.0) == '1.0/x'
    assert ccode(x**Rational(2, 3)) == 'pow(x, 2.0L/3.0L)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    assert ccode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert ccode(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 3.2)'


def test_ccode_constants_mathh():
    assert ccode(exp(1)) == "M_E"
    assert ccode(pi) == "M_PI"
    assert ccode(oo) == "HUGE_VAL"
    assert ccode(-oo) == "-HUGE_VAL"


def test_ccode_constants_other():
    assert ccode(2*GoldenRatio) == "2*GoldenRatio"
    assert ccode(2*Catalan) == "2*Catalan"
    assert ccode(2*EulerGamma) == "2*EulerGamma"


def test_ccode_Rational():
    assert ccode(Rational(3, 7)) == "3.0L/7.0L"
    assert ccode(Rational(18, 9)) == "2"
    assert ccode(Rational(3, -7)) == "-3.0L/7.0L"
    assert ccode(Rational(-3, -7)) == "3.0L/7.0L"
    assert ccode(x + Rational(3, 7)) == "x + 3.0L/7.0L"
    assert ccode(Rational(3, 7)*x) == "(3.0L/7.0L)*x"


def test_ccode_Integer():
    assert ccode(Integer(67)) == "67"
    assert ccode(Integer(-1)) == "-1"


def test_ccode_functions():
    assert ccode(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"
    assert ccode(ceiling(x)) == "ceil(x)"
    assert ccode(Abs(x)) == "fabs(x)"
    assert ccode(gamma(x)) == "tgamma(x)"


def test_ccode_inline_function():
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(g(x)) == "2*x"


def test_ccode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    }
    assert ccode(ceiling(x), user_functions=custom_functions) == "ceil(x)"
    assert ccode(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert ccode(Abs(n), user_functions=custom_functions) == "abs(n)"


def test_ccode_boolean():
    assert ccode(x & y) == "x && y"
    assert ccode(x | y) == "x || y"
    assert ccode(~x) == "!x"
    assert ccode(x & y & z) == "x && y && z"
    assert ccode(x | y | z) == "x || y || z"
    assert ccode((x & y) | z) == "z || x && y"
    assert ccode((x | y) & z) == "z && (x || y)"


def test_ccode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "    x\n"
            ")\n"
            ": (\n"
            "    pow(x, 2)\n"
            "))")
    expr = Piecewise((Assign(c, x), x < 1), (Assign(c, x**2), True))
    assert ccode(expr) == (
            "if (x < 1) {\n"
            "    c = x;\n"
            "}\n"
            "else {\n"
            "    c = pow(x, 2);\n"
            "}")
    expr = Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True))
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "    x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "    x + 1\n"
            ")\n"
            ": (\n"
            "    pow(x, 2)\n"
            ")))")
    expr = Piecewise((Assign(c, x), x < 1), (Assign(c, x + 1), x < 2), (Assign(c, x**2), True))
    assert ccode(expr) == (
            "if (x < 1) {\n"
            "    c = x;\n"
            "}\n"
            "else if (x < 2) {\n"
            "    c = x + 1;\n"
            "}\n"
            "else {\n"
            "    c = pow(x, 2);\n"
            "}")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: ccode(expr))


def test_ccode_Piecewise_deep():
    p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    assert p == (
            "2*((x < 1) ? (\n"
            "    x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "    x + 1\n"
            ")\n"
            ": (\n"
            "    pow(x, 2)\n"
            ")))")
    expr = x*y*z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    assert ccode(expr) == (
            "pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "    0\n"
            ")\n"
            ": (\n"
            "    1\n"
            ")) + cos(z) - 1")
    assert ccode(expr, assign_to='c') == (
            "c = pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "    0\n"
            ")\n"
            ": (\n"
            "    1\n"
            ")) + cos(z) - 1;")


def test_ccode_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    x = IndexedBase('x')[j]
    assert ccode(x) == 'x[j]'
    A = IndexedBase('A')[i, j]
    assert ccode(A) == 'A[%s]' % (m*i+j)
    B = IndexedBase('B')[i, j, k]
    assert ccode(B) == 'B[%s]' % (i*o*m+j*o+k)


def test_ccode_settings():
    raises(TypeError, lambda: ccode(sin(x), method="garbage"))


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert ccode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


def test_ccode_Assign():
    assert ccode(Assign(x, y + z)) == 'x = y + z;'
    assert ccode(AugAssign(x, '+', y + z)) == 'x += y + z;'


def test_ccode_For():
    f = For(x, Range(0, 10, 2), [AugAssign(y, '*', x)])
    sol = ccode(f)
    assert sol == ("for (x = 0; x < 10; x += 2) {\n"
                   "    y *= x;\n"
                   "}")


def test_ccode_FunctionDef():
    name = 'test'
    args = (InArgument('double', a), InArgument('int', b))
    body = (Return(sin(a) + cos(b)),)
    results = (Result('double'),)
    f = FunctionDef(name, args, body, results)
    assert ccode(f) == ("double test(double a, int b) {\n"
                        "    return sin(a) + cos(b);\n"
                        "}")


def test_ccode_Import():
    assert ccode(Import('math.h')) == '#include "math.h"'


def test_ccode_Declare():
    assert ccode(Declare('int', Variable('int', a))) == 'int a;'
    assert ccode(Declare('double', (Variable('double', a), Variable('double', b)))) == 'double a, b;'
