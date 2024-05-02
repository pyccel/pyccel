# pylint: disable=missing-function-docstring, missing-module-docstring/
from sympy import (sin, cos, atan2, log, exp, gamma, conjugate, sqrt,
        factorial, Piecewise,  symbols, S, Float)
from sympy import Catalan, EulerGamma, GoldenRatio, I
from sympy import Function, Rational, Integer, Lambda

from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.lambdify import implemented_function
from sympy.sets.fancysets import Range
from sympy.utilities.pytest import raises

from pyccel.types.ast import Assign, For, Import, Declare, InArgument, InOutArgument, OutArgument
from pyccel.printers import fcode, FCodePrinter

x, y, z = symbols('x, y, z')
a, b, c = symbols('a, b, c')
n = symbols('n', integer=True)
m = symbols('m', imaginary=True)


def test_printmethod():
    class nint(Function):
        def _fcode(self, printer):
            return "nint(%s)" % printer._print(self.args[0])
    assert fcode(nint(x)) == "nint(x)"


def test_fcode_sqrt():
    assert fcode(sqrt(x)) == 'sqrt(x)'
    assert fcode(sqrt(n)) == 'sqrt(dble(n))'
    assert fcode(x**0.5) == 'sqrt(x)'
    assert fcode(sqrt(10)) == 'sqrt(10.0d0)'


def test_fcode_Pow():
    assert fcode(x**3) == "x**3"
    assert fcode(x**(y**3)) == "x**(y**3)"
    g = implemented_function('g2', Lambda(x, sin(x)))
    assert fcode((g(x)*3.5)**(x - y**x)/(x**2 + y)) == "(3.5d0*sin(x))**(x - y**x)/(x**2 + y)"
    assert fcode(x**-1.0) == '1.0/x'
    assert fcode(x**Rational(2, 3)) == 'x**(2.0d0/3.0d0)'
    assert fcode(x**-2.0, 'y') == 'y = x**(-2.0d0)'


def test_fcode_constants_other():
    assert fcode(2*GoldenRatio) == "2*GoldenRatio"
    assert fcode(2*Catalan) == "2*Catalan"
    assert fcode(2*EulerGamma) == "2*EulerGamma"


def test_fcode_Rational():
    x = symbols('x')
    assert fcode(Rational(3, 7)) == "3.0d0/7.0d0"
    assert fcode(Rational(18, 9)) == "2"
    assert fcode(Rational(3, -7)) == "-3.0d0/7.0d0"
    assert fcode(Rational(-3, -7)) == "3.0d0/7.0d0"
    assert fcode(x + Rational(3, 7)) == "x + 3.0d0/7.0d0"
    assert fcode(Rational(3, 7)*x) == "(3.0d0/7.0d0)*x"


def test_fcode_Integer():
    assert fcode(Integer(67)) == "67"
    assert fcode(Integer(-1)) == "-1"


def test_fcode_Float():
    assert fcode(Float(42.0)) == "42.0000000000000d0"
    assert fcode(Float(-1e20)) == "-1.00000000000000d+20"


def test_fcode_complex():
    assert fcode(I) == "cmplx(0,1)"
    assert fcode(4*I) == "cmplx(0,4)"
    assert fcode(3 + 4*I) == "cmplx(3,4)"
    assert fcode(3 + 4*I + x) == "cmplx(3,4) + x"
    assert fcode(I*x) == "cmplx(0,1)*x"
    assert fcode(3 + 4*I - x) == "cmplx(3,4) - x"
    assert fcode(5*m) == "5*m"
    assert fcode(I*m) == "cmplx(0,1)*m"
    assert fcode(3 + m) == "m + 3"


def test_fcode_functions():
    assert fcode(sin(x) ** cos(y)) == "sin(x)**cos(y)"
    assert fcode(sin(x)) == "sin(x)"
    assert fcode(atan2(x, y)) == "atan2(x, y)"
    assert fcode(conjugate(x)) == "conjg(x)"


#issue 6814
def test_fcode_functions_pre_evaluate():
    assert fcode(x * log(10)) == "x*2.30258509299405d0"
    assert fcode(x * log(10)) == "x*2.30258509299405d0"
    assert fcode(x * log(S(10))) == "x*2.30258509299405d0"
    assert fcode(log(S(10))) == "2.30258509299405d0"
    assert fcode(exp(10)) == "22026.4657948067d0"
    assert fcode(x * log(log(10))) == "x*0.834032445247956d0"
    assert fcode(x * log(log(S(10)))) == "x*0.834032445247956d0"


def test_inline_function():
    g = implemented_function('g', Lambda(x, 2*x))
    assert fcode(g(x)) == "2*x"


def test_user_functions():
    g = Function('g')
    assert fcode(g(x), user_functions={"g": "great"}) == "great(x)"
    assert fcode(sin(x), user_functions={"sin": "zsin"}) == "zsin(x)"
    assert fcode(gamma(x), user_functions={"gamma": "mygamma"}) == "mygamma(x)"
    assert fcode(factorial(n), user_functions={"factorial": "fct"}) == "fct(n)"


def test_fcode_Logical():
    # unary Not
    assert fcode(Not(x)) == ".not. x"
    # binary And
    assert fcode(And(x, y)) == "x .and. y"
    assert fcode(And(x, Not(y))) == "x .and. .not. y"
    assert fcode(And(Not(x), y)) == "y .and. .not. x"
    assert fcode(And(Not(x), Not(y))) ==  ".not. x .and. .not. y"
    assert fcode(Not(And(x, y), evaluate=False)) == ".not. (x .and. y)"
    # binary Or
    assert fcode(Or(x, y)) == "x .or. y"
    assert fcode(Or(x, Not(y))) == "x .or. .not. y"
    assert fcode(Or(Not(x), y)) == "y .or. .not. x"
    assert fcode(Or(Not(x), Not(y))) == ".not. x .or. .not. y"
    assert fcode(Not(Or(x, y), evaluate=False)) == ".not. (x .or. y)"
    # mixed And/Or
    assert fcode(And(Or(y, z), x)) == "x .and. (y .or. z)"
    assert fcode(And(Or(z, x), y)) == "y .and. (x .or. z)"
    assert fcode(And(Or(x, y), z)) == "z .and. (x .or. y)"
    assert fcode(Or(And(y, z), x)) == "x .or. y .and. z"
    assert fcode(Or(And(z, x), y)) == "y .or. x .and. z"
    assert fcode(Or(And(x, y), z)) == "z .or. x .and. y"
    # trinary And
    assert fcode(And(x, y, z)) == "x .and. y .and. z"
    assert fcode(And(x, y, Not(z))) == "x .and. y .and. .not. z"
    assert fcode(And(x, Not(y), z)) == "x .and. z .and. .not. y"
    assert fcode(And(Not(x), y, z)) == "y .and. z .and. .not. x"
    assert fcode(Not(And(x, y, z), evaluate=False)) == ".not. (x .and. y .and. z)"
    # trinary Or
    assert fcode(Or(x, y, z)) == "x .or. y .or. z"
    assert fcode(Or(x, y, Not(z))) == "x .or. y .or. .not. z"
    assert fcode(Or(x, Not(y), z)) == "x .or. z .or. .not. y"
    assert fcode(Or(Not(x), y, z)) == "y .or. z .or. .not. x"
    assert fcode(Not(Or(x, y, z), evaluate=False)) == ".not. (x .or. y .or. z)"


def test_fcode_Xlogical():
    x, y, z = symbols("x y z")
    # binary Xor
    assert fcode(Xor(x, y, evaluate=False)) == "x .neqv. y"
    assert fcode(Xor(x, Not(y), evaluate=False)) == "x .neqv. .not. y"
    assert fcode(Xor(Not(x), y, evaluate=False)) == "y .neqv. .not. x"
    assert fcode(Xor(Not(x), Not(y), evaluate=False)) == ".not. x .neqv. .not. y"
    assert fcode(Not(Xor(x, y, evaluate=False), evaluate=False)) == ".not. (x .neqv. y)"
    # binary Equivalent
    assert fcode(Equivalent(x, y)) == "x .eqv. y"
    assert fcode(Equivalent(x, Not(y))) == "x .eqv. .not. y"
    assert fcode(Equivalent(Not(x), y)) == "y .eqv. .not. x"
    assert fcode(Equivalent(Not(x), Not(y))) == ".not. x .eqv. .not. y"
    assert fcode(Not(Equivalent(x, y), evaluate=False)) == ".not. (x .eqv. y)"
    # mixed And/Equivalent
    assert fcode(Equivalent(And(y, z), x)) == "x .eqv. y .and. z"
    assert fcode(Equivalent(And(z, x), y)) == "y .eqv. x .and. z"
    assert fcode(Equivalent(And(x, y), z)) == "z .eqv. x .and. y"
    assert fcode(And(Equivalent(y, z), x)) == "x .and. (y .eqv. z)"
    assert fcode(And(Equivalent(z, x), y)) == "y .and. (x .eqv. z)"
    assert fcode(And(Equivalent(x, y), z)) == "z .and. (x .eqv. y)"
    # mixed Or/Equivalent
    assert fcode(Equivalent(Or(y, z), x)) == "x .eqv. y .or. z"
    assert fcode(Equivalent(Or(z, x), y)) == "y .eqv. x .or. z"
    assert fcode(Equivalent(Or(x, y), z)) == "z .eqv. x .or. y"
    assert fcode(Or(Equivalent(y, z), x)) == "x .or. (y .eqv. z)"
    assert fcode(Or(Equivalent(z, x), y)) == "y .or. (x .eqv. z)"
    assert fcode(Or(Equivalent(x, y), z)) == "z .or. (x .eqv. y)"
    # mixed Xor/Equivalent
    assert fcode(Equivalent(Xor(y, z, evaluate=False), x)) == "x .eqv. (y .neqv. z)"
    assert fcode(Equivalent(Xor(z, x, evaluate=False), y)) == "y .eqv. (x .neqv. z)"
    assert fcode(Equivalent(Xor(x, y, evaluate=False), z)) == "z .eqv. (x .neqv. y)"
    assert fcode(Xor(Equivalent(y, z), x, evaluate=False)) == "x .neqv. (y .eqv. z)"
    assert fcode(Xor(Equivalent(z, x), y, evaluate=False)) == "y .neqv. (x .eqv. z)"
    assert fcode(Xor(Equivalent(x, y), z, evaluate=False)) == "z .neqv. (x .eqv. y)"
    # mixed And/Xor
    assert fcode(Xor(And(y, z), x, evaluate=False)) == "x .neqv. y .and. z"
    assert fcode(Xor(And(z, x), y, evaluate=False)) == "y .neqv. x .and. z"
    assert fcode(Xor(And(x, y), z, evaluate=False)) == "z .neqv. x .and. y"
    assert fcode(And(Xor(y, z, evaluate=False), x)) == "x .and. (y .neqv. z)"
    assert fcode(And(Xor(z, x, evaluate=False), y)) == "y .and. (x .neqv. z)"
    assert fcode(And(Xor(x, y, evaluate=False), z)) == "z .and. (x .neqv. y)"
    # mixed Or/Xor
    assert fcode(Xor(Or(y, z), x, evaluate=False)) == "x .neqv. y .or. z"
    assert fcode(Xor(Or(z, x), y, evaluate=False)) == "y .neqv. x .or. z"
    assert fcode(Xor(Or(x, y), z, evaluate=False)) == "z .neqv. x .or. y"
    assert fcode(Or(Xor(y, z, evaluate=False), x)) == "x .or. (y .neqv. z)"
    assert fcode(Or(Xor(z, x, evaluate=False), y)) == "y .or. (x .neqv. z)"
    assert fcode(Or(Xor(x, y, evaluate=False), z)) == "z .or. (x .neqv. y)"
    # ternary Xor
    assert fcode(Xor(x, y, z, evaluate=False)) == "x .neqv. y .neqv. z"
    assert fcode(Xor(x, y, Not(z), evaluate=False)) == "x .neqv. y .neqv. .not. z"
    assert fcode(Xor(x, Not(y), z, evaluate=False)) == "x .neqv. z .neqv. .not. y"
    assert fcode(Xor(Not(x), y, z, evaluate=False)) == "y .neqv. z .neqv. .not. x"


# TODO not working even using sympy
#def test_fcode_Relational():
#    assert fcode(Relational(x, y, "==")) == "x == y"
#    assert fcode(Relational(x, y, "!=")) == "x /= y"
#    assert fcode(Relational(x, y, ">=")) == "x >= y"
#    assert fcode(Relational(x, y, "<=")) == "x <= y"
#    assert fcode(Relational(x, y, ">")) == "x > y"
#    assert fcode(Relational(x, y, "<")) == "x < y"


def test_fcode_precedence():
    assert fcode(And(x < y, y < x + 1)) == "x < y .and. y < x + 1"
    assert fcode(Or(x < y, y < x + 1)) == "x < y .or. y < x + 1"
    assert fcode(Xor(x < y, y < x + 1, evaluate=False)) == "x < y .neqv. y < x + 1"
    assert fcode(Equivalent(x < y, y < x + 1)) == "x < y .eqv. y < x + 1"


def test_fcode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    assert fcode(expr) == "merge(x, x**2, x < 1)"
    expr = Piecewise((Assign(c, x), x < 1), (Assign(c, x**2), True))
    assert fcode(expr) == (
            "if (x < 1) then\n"
            "    c = x\n"
            "else\n"
            "    c = x**2\n"
            "end if"
    )
    expr = Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True))
    assert fcode(expr) == "merge(x, merge(x + 1, x**2, x < 2), x < 1)"
    expr = Piecewise((Assign(c, x), x < 1), (Assign(c, x + 1), x < 2), (Assign(c, x**2), True))
    assert fcode(expr) == (
            "if (x < 1) then\n"
            "    c = x\n"
            "else if (x < 2) then\n"
            "    c = x + 1\n"
            "else\n"
            "    c = x**2\n"
            "end if")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: fcode(expr))


def test_fcode_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    x = IndexedBase('x')[j]
    assert fcode(x) == 'x(j)'
    A = IndexedBase('A')[i, j]
    assert fcode(A) == 'A(i, j)'
    B = IndexedBase('B')[i, j, k]
    assert fcode(B) == 'B(i, j, k)'


def test_fcode_settings():
    raises(TypeError, lambda: fcode(S(4), method="garbage"))


def test_fcode_continuation_line():
    x, y = symbols('x,y')
    result = fcode(((cos(x) + sin(y))**(7)).expand())
    expected = (
        'sin(y)**7 + 7*sin(y)**6*cos(x) + 21*sin(y)**5*cos(x)**2 + 35*sin(y)**4* &\n'
        '      cos(x)**3 + 35*sin(y)**3*cos(x)**4 + 21*sin(y)**2*cos(x)**5 + 7* &\n'
        '      sin(y)*cos(x)**6 + cos(x)**7'
    )
    assert result == expected


def test_fcode_comment_line():
    printer = FCodePrinter()
    lines = [ "! This is a long comment on a single line that must be wrapped properly to produce nice output"]
    expected = [
        '! This is a long comment on a single line that must be wrapped properly',
        '! to produce nice output']
    assert printer._wrap_fortran(lines) == expected


def test_indent():
    codelines = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        'do \n'
        'do j = 1, 5\n'
        'if (a>b) then\n'
        'if(b>0) then\n'
        'a = 3\n'
        'donot_indent_me = 2\n'
        'do_not_indent_me_either = 2\n'
        'ifIam_indented_something_went_wrong = 2\n'
        'if_I_am_indented_something_went_wrong = 2\n'
        'end should not be unindented here\n'
        'end if\n'
        'endif\n'
        'end do\n'
        'end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        'a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    expected = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        '    do \n'
        '        do j = 1, 5\n'
        '            if (a>b) then\n'
        '                if(b>0) then\n'
        '                    a = 3\n'
        '                    donot_indent_me = 2\n'
        '                    do_not_indent_me_either = 2\n'
        '                    ifIam_indented_something_went_wrong = 2\n'
        '                    if_I_am_indented_something_went_wrong = 2\n'
        '                    end should not be unindented here\n'
        '                end if\n'
        '            endif\n'
        '        end do\n'
        '    end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        '    a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    p = FCodePrinter()
    result = p.indent_code(codelines)
    assert result == expected


def test_fcode_Assign():
    assert fcode(Assign(x, y + z)) == 'x = y + z'


def test_fcode_For():
    f = For(x, Range(0, 10, 2), [Assign(y, x * y)])
    sol = fcode(f)
    assert sol == ("do x = 0, 10, 2\n"
                   "    y = x*y\n"
                   "end do")


def test_fcode_Import():
    assert fcode(Import('math', 'sin')) == 'use math, only: sin'


def test_fcode_Declare():
    assert fcode(Declare('int', InArgument('int', a))) == "integer, intent(in) :: a"
    assert fcode(Declare('double', (InArgument('double', a), InArgument('double',
            b), OutArgument('double', c),
            InOutArgument('double', x)))) == ("real(dp), intent(in) :: a, b\n"
                                              "real(dp), intent(inout) :: x\n"
                                              "real(dp), intent(out) :: c")

#####################################################
if __name__ == "__main__":
    test_printmethod()
    test_fcode_sqrt()
