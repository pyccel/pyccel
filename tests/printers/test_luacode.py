# pylint: disable=missing-function-docstring, missing-module-docstring/
from sympy import (sin, cos, atan2, log, exp, gamma, conjugate, sqrt,
        factorial, Piecewise,  symbols, S, Float)
from sympy import Catalan, EulerGamma, GoldenRatio, I
from sympy import Function, Rational, Integer, Lambda

from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.lambdify import implemented_function
from sympy.sets.fancysets import Range
from sympy.utilities.pytest import raises

from pyccel.types.ast import Assign, For, Import, Declare, Variable, InArgument, InOutArgument, OutArgument
from pyccel.printers import lua_code

x, y, z = symbols('x, y, z')
a, b, c = symbols('a, b, c')
n = symbols('n', integer=True)
m = symbols('m', imaginary=True)


#def test_printmethod():
#    class nint(Function):
#        def _lua_code(self, printer):
#            return "nint(%s)" % printer._print(self.args[0])
#    assert lua_code(nint(x)) == "nint(x)"
#
#
#def test_lua_code_sqrt():
#    assert lua_code(sqrt(x)) == 'sqrt(x)'
#    assert lua_code(sqrt(n)) == 'sqrt(dble(n))'
#    assert lua_code(x**0.5) == 'sqrt(x)'
#    assert lua_code(sqrt(10)) == 'sqrt(10.0d0)'


def test_lua_code_Pow():
    print(lua_code(x**3))
#    assert lua_code(x**3) == "pow(x,3)"
#    assert lua_code(x**(y**3)) == "pow(x, pow(y,3))"
#    g = implemented_function('g2', Lambda(x, sin(x)))
#    assert lua_code((g(x)*3.5)**(x - y**x)/(x**2 + y)) == "(3.5d0*sin(x))**(x - y**x)/(x**2 + y)"
#    assert lua_code(x**-1.0) == '1.0/x'
#    assert lua_code(x**Rational(2, 3)) == 'x**(2.0d0/3.0d0)'
#    assert lua_code(x**-2.0, 'y') == 'y = x**(-2.0d0)'


#def test_lua_code_constants_other():
#    assert lua_code(2*GoldenRatio) == "2*GoldenRatio"
#    assert lua_code(2*Catalan) == "2*Catalan"
#    assert lua_code(2*EulerGamma) == "2*EulerGamma"
#
#
#def test_lua_code_Rational():
#    x = symbols('x')
#    assert lua_code(Rational(3, 7)) == "3.0d0/7.0d0"
#    assert lua_code(Rational(18, 9)) == "2"
#    assert lua_code(Rational(3, -7)) == "-3.0d0/7.0d0"
#    assert lua_code(Rational(-3, -7)) == "3.0d0/7.0d0"
#    assert lua_code(x + Rational(3, 7)) == "x + 3.0d0/7.0d0"
#    assert lua_code(Rational(3, 7)*x) == "(3.0d0/7.0d0)*x"
#
#
#def test_lua_code_Integer():
#    assert lua_code(Integer(67)) == "67"
#    assert lua_code(Integer(-1)) == "-1"
#
#
#def test_lua_code_Float():
#    assert lua_code(Float(42.0)) == "42.0000000000000d0"
#    assert lua_code(Float(-1e20)) == "-1.00000000000000d+20"
#
#
#def test_lua_code_complex():
#    assert lua_code(I) == "cmplx(0,1)"
#    assert lua_code(4*I) == "cmplx(0,4)"
#    assert lua_code(3 + 4*I) == "cmplx(3,4)"
#    assert lua_code(3 + 4*I + x) == "cmplx(3,4) + x"
#    assert lua_code(I*x) == "cmplx(0,1)*x"
#    assert lua_code(3 + 4*I - x) == "cmplx(3,4) - x"
#    assert lua_code(5*m) == "5*m"
#    assert lua_code(I*m) == "cmplx(0,1)*m"
#    assert lua_code(3 + m) == "m + 3"
#
#
#def test_lua_code_functions():
#    assert lua_code(sin(x) ** cos(y)) == "sin(x)**cos(y)"
#    assert lua_code(sin(x)) == "sin(x)"
#    assert lua_code(atan2(x, y)) == "atan2(x, y)"
#    assert lua_code(conjugate(x)) == "conjg(x)"
#
#
##issue 6814
#def test_lua_code_functions_pre_evaluate():
#    assert lua_code(x * log(10)) == "x*2.30258509299405d0"
#    assert lua_code(x * log(10)) == "x*2.30258509299405d0"
#    assert lua_code(x * log(S(10))) == "x*2.30258509299405d0"
#    assert lua_code(log(S(10))) == "2.30258509299405d0"
#    assert lua_code(exp(10)) == "22026.4657948067d0"
#    assert lua_code(x * log(log(10))) == "x*0.834032445247956d0"
#    assert lua_code(x * log(log(S(10)))) == "x*0.834032445247956d0"
#
#
#def test_inline_function():
#    g = implemented_function('g', Lambda(x, 2*x))
#    assert lua_code(g(x)) == "2*x"
#
#
#def test_user_functions():
#    g = Function('g')
#    assert lua_code(g(x), user_functions={"g": "great"}) == "great(x)"
#    assert lua_code(sin(x), user_functions={"sin": "zsin"}) == "zsin(x)"
#    assert lua_code(gamma(x), user_functions={"gamma": "mygamma"}) == "mygamma(x)"
#    assert lua_code(factorial(n), user_functions={"factorial": "fct"}) == "fct(n)"
#
#
#def test_lua_code_Logical():
#    # unary Not
#    assert lua_code(Not(x)) == ".not. x"
#    # binary And
#    assert lua_code(And(x, y)) == "x .and. y"
#    assert lua_code(And(x, Not(y))) == "x .and. .not. y"
#    assert lua_code(And(Not(x), y)) == "y .and. .not. x"
#    assert lua_code(And(Not(x), Not(y))) ==  ".not. x .and. .not. y"
#    assert lua_code(Not(And(x, y), evaluate=False)) == ".not. (x .and. y)"
#    # binary Or
#    assert lua_code(Or(x, y)) == "x .or. y"
#    assert lua_code(Or(x, Not(y))) == "x .or. .not. y"
#    assert lua_code(Or(Not(x), y)) == "y .or. .not. x"
#    assert lua_code(Or(Not(x), Not(y))) == ".not. x .or. .not. y"
#    assert lua_code(Not(Or(x, y), evaluate=False)) == ".not. (x .or. y)"
#    # mixed And/Or
#    assert lua_code(And(Or(y, z), x)) == "x .and. (y .or. z)"
#    assert lua_code(And(Or(z, x), y)) == "y .and. (x .or. z)"
#    assert lua_code(And(Or(x, y), z)) == "z .and. (x .or. y)"
#    assert lua_code(Or(And(y, z), x)) == "x .or. y .and. z"
#    assert lua_code(Or(And(z, x), y)) == "y .or. x .and. z"
#    assert lua_code(Or(And(x, y), z)) == "z .or. x .and. y"
#    # trinary And
#    assert lua_code(And(x, y, z)) == "x .and. y .and. z"
#    assert lua_code(And(x, y, Not(z))) == "x .and. y .and. .not. z"
#    assert lua_code(And(x, Not(y), z)) == "x .and. z .and. .not. y"
#    assert lua_code(And(Not(x), y, z)) == "y .and. z .and. .not. x"
#    assert lua_code(Not(And(x, y, z), evaluate=False)) == ".not. (x .and. y .and. z)"
#    # trinary Or
#    assert lua_code(Or(x, y, z)) == "x .or. y .or. z"
#    assert lua_code(Or(x, y, Not(z))) == "x .or. y .or. .not. z"
#    assert lua_code(Or(x, Not(y), z)) == "x .or. z .or. .not. y"
#    assert lua_code(Or(Not(x), y, z)) == "y .or. z .or. .not. x"
#    assert lua_code(Not(Or(x, y, z), evaluate=False)) == ".not. (x .or. y .or. z)"
#
#
#def test_lua_code_Xlogical():
#    x, y, z = symbols("x y z")
#    # binary Xor
#    assert lua_code(Xor(x, y, evaluate=False)) == "x .neqv. y"
#    assert lua_code(Xor(x, Not(y), evaluate=False)) == "x .neqv. .not. y"
#    assert lua_code(Xor(Not(x), y, evaluate=False)) == "y .neqv. .not. x"
#    assert lua_code(Xor(Not(x), Not(y), evaluate=False)) == ".not. x .neqv. .not. y"
#    assert lua_code(Not(Xor(x, y, evaluate=False), evaluate=False)) == ".not. (x .neqv. y)"
#    # binary Equivalent
#    assert lua_code(Equivalent(x, y)) == "x .eqv. y"
#    assert lua_code(Equivalent(x, Not(y))) == "x .eqv. .not. y"
#    assert lua_code(Equivalent(Not(x), y)) == "y .eqv. .not. x"
#    assert lua_code(Equivalent(Not(x), Not(y))) == ".not. x .eqv. .not. y"
#    assert lua_code(Not(Equivalent(x, y), evaluate=False)) == ".not. (x .eqv. y)"
#    # mixed And/Equivalent
#    assert lua_code(Equivalent(And(y, z), x)) == "x .eqv. y .and. z"
#    assert lua_code(Equivalent(And(z, x), y)) == "y .eqv. x .and. z"
#    assert lua_code(Equivalent(And(x, y), z)) == "z .eqv. x .and. y"
#    assert lua_code(And(Equivalent(y, z), x)) == "x .and. (y .eqv. z)"
#    assert lua_code(And(Equivalent(z, x), y)) == "y .and. (x .eqv. z)"
#    assert lua_code(And(Equivalent(x, y), z)) == "z .and. (x .eqv. y)"
#    # mixed Or/Equivalent
#    assert lua_code(Equivalent(Or(y, z), x)) == "x .eqv. y .or. z"
#    assert lua_code(Equivalent(Or(z, x), y)) == "y .eqv. x .or. z"
#    assert lua_code(Equivalent(Or(x, y), z)) == "z .eqv. x .or. y"
#    assert lua_code(Or(Equivalent(y, z), x)) == "x .or. (y .eqv. z)"
#    assert lua_code(Or(Equivalent(z, x), y)) == "y .or. (x .eqv. z)"
#    assert lua_code(Or(Equivalent(x, y), z)) == "z .or. (x .eqv. y)"
#    # mixed Xor/Equivalent
#    assert lua_code(Equivalent(Xor(y, z, evaluate=False), x)) == "x .eqv. (y .neqv. z)"
#    assert lua_code(Equivalent(Xor(z, x, evaluate=False), y)) == "y .eqv. (x .neqv. z)"
#    assert lua_code(Equivalent(Xor(x, y, evaluate=False), z)) == "z .eqv. (x .neqv. y)"
#    assert lua_code(Xor(Equivalent(y, z), x, evaluate=False)) == "x .neqv. (y .eqv. z)"
#    assert lua_code(Xor(Equivalent(z, x), y, evaluate=False)) == "y .neqv. (x .eqv. z)"
#    assert lua_code(Xor(Equivalent(x, y), z, evaluate=False)) == "z .neqv. (x .eqv. y)"
#    # mixed And/Xor
#    assert lua_code(Xor(And(y, z), x, evaluate=False)) == "x .neqv. y .and. z"
#    assert lua_code(Xor(And(z, x), y, evaluate=False)) == "y .neqv. x .and. z"
#    assert lua_code(Xor(And(x, y), z, evaluate=False)) == "z .neqv. x .and. y"
#    assert lua_code(And(Xor(y, z, evaluate=False), x)) == "x .and. (y .neqv. z)"
#    assert lua_code(And(Xor(z, x, evaluate=False), y)) == "y .and. (x .neqv. z)"
#    assert lua_code(And(Xor(x, y, evaluate=False), z)) == "z .and. (x .neqv. y)"
#    # mixed Or/Xor
#    assert lua_code(Xor(Or(y, z), x, evaluate=False)) == "x .neqv. y .or. z"
#    assert lua_code(Xor(Or(z, x), y, evaluate=False)) == "y .neqv. x .or. z"
#    assert lua_code(Xor(Or(x, y), z, evaluate=False)) == "z .neqv. x .or. y"
#    assert lua_code(Or(Xor(y, z, evaluate=False), x)) == "x .or. (y .neqv. z)"
#    assert lua_code(Or(Xor(z, x, evaluate=False), y)) == "y .or. (x .neqv. z)"
#    assert lua_code(Or(Xor(x, y, evaluate=False), z)) == "z .or. (x .neqv. y)"
#    # ternary Xor
#    assert lua_code(Xor(x, y, z, evaluate=False)) == "x .neqv. y .neqv. z"
#    assert lua_code(Xor(x, y, Not(z), evaluate=False)) == "x .neqv. y .neqv. .not. z"
#    assert lua_code(Xor(x, Not(y), z, evaluate=False)) == "x .neqv. z .neqv. .not. y"
#    assert lua_code(Xor(Not(x), y, z, evaluate=False)) == "y .neqv. z .neqv. .not. x"
#
#
## TODO not working even using sympy
##def test_lua_code_Relational():
##    assert lua_code(Relational(x, y, "==")) == "x == y"
##    assert lua_code(Relational(x, y, "!=")) == "x /= y"
##    assert lua_code(Relational(x, y, ">=")) == "x >= y"
##    assert lua_code(Relational(x, y, "<=")) == "x <= y"
##    assert lua_code(Relational(x, y, ">")) == "x > y"
##    assert lua_code(Relational(x, y, "<")) == "x < y"
#
#
#def test_lua_code_precedence():
#    assert lua_code(And(x < y, y < x + 1)) == "x < y .and. y < x + 1"
#    assert lua_code(Or(x < y, y < x + 1)) == "x < y .or. y < x + 1"
#    assert lua_code(Xor(x < y, y < x + 1, evaluate=False)) == "x < y .neqv. y < x + 1"
#    assert lua_code(Equivalent(x < y, y < x + 1)) == "x < y .eqv. y < x + 1"
#
#
#def test_lua_code_Piecewise():
#    expr = Piecewise((x, x < 1), (x**2, True))
#    assert lua_code(expr) == "merge(x, x**2, x < 1)"
#    expr = Piecewise((Assign(c, x), x < 1), (Assign(c, x**2), True))
#    assert lua_code(expr) == (
#            "if (x < 1) then\n"
#            "    c = x\n"
#            "else\n"
#            "    c = x**2\n"
#            "end if"
#    )
#    expr = Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True))
#    assert lua_code(expr) == "merge(x, merge(x + 1, x**2, x < 2), x < 1)"
#    expr = Piecewise((Assign(c, x), x < 1), (Assign(c, x + 1), x < 2), (Assign(c, x**2), True))
#    assert lua_code(expr) == (
#            "if (x < 1) then\n"
#            "    c = x\n"
#            "else if (x < 2) then\n"
#            "    c = x + 1\n"
#            "else\n"
#            "    c = x**2\n"
#            "end if")
#    # Check that Piecewise without a True (default) condition error
#    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
#    raises(ValueError, lambda: lua_code(expr))
#
#
#def test_lua_code_Indexed():
#    n, m, o = symbols('n m o', integer=True)
#    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
#    x = IndexedBase('x')[j]
#    assert lua_code(x) == 'x(j)'
#    A = IndexedBase('A')[i, j]
#    assert lua_code(A) == 'A(i, j)'
#    B = IndexedBase('B')[i, j, k]
#    assert lua_code(B) == 'B(i, j, k)'
#
#
#def test_lua_code_settings():
#    raises(TypeError, lambda: lua_code(S(4), method="garbage"))
#
#
#def test_lua_code_continuation_line():
#    x, y = symbols('x,y')
#    result = lua_code(((cos(x) + sin(y))**(7)).expand())
#    expected = (
#        'sin(y)**7 + 7*sin(y)**6*cos(x) + 21*sin(y)**5*cos(x)**2 + 35*sin(y)**4* &\n'
#        '      cos(x)**3 + 35*sin(y)**3*cos(x)**4 + 21*sin(y)**2*cos(x)**5 + 7* &\n'
#        '      sin(y)*cos(x)**6 + cos(x)**7'
#    )
#    assert result == expected
#
#
#def test_lua_code_comment_line():
#    printer = FCodePrinter()
#    lines = [ "! This is a long comment on a single line that must be wrapped properly to produce nice output"]
#    expected = [
#        '! This is a long comment on a single line that must be wrapped properly',
#        '! to produce nice output']
#    assert printer._wrap_fortran(lines) == expected
#
#
#def test_indent():
#    codelines = (
#        'subroutine test(a)\n'
#        'integer :: a, i, j\n'
#        '\n'
#        'do\n'
#        'do \n'
#        'do j = 1, 5\n'
#        'if (a>b) then\n'
#        'if(b>0) then\n'
#        'a = 3\n'
#        'donot_indent_me = 2\n'
#        'do_not_indent_me_either = 2\n'
#        'ifIam_indented_something_went_wrong = 2\n'
#        'if_I_am_indented_something_went_wrong = 2\n'
#        'end should not be unindented here\n'
#        'end if\n'
#        'endif\n'
#        'end do\n'
#        'end do\n'
#        'enddo\n'
#        'end subroutine\n'
#        '\n'
#        'subroutine test2(a)\n'
#        'integer :: a\n'
#        'do\n'
#        'a = a + 1\n'
#        'end do \n'
#        'end subroutine\n'
#    )
#    expected = (
#        'subroutine test(a)\n'
#        'integer :: a, i, j\n'
#        '\n'
#        'do\n'
#        '    do \n'
#        '        do j = 1, 5\n'
#        '            if (a>b) then\n'
#        '                if(b>0) then\n'
#        '                    a = 3\n'
#        '                    donot_indent_me = 2\n'
#        '                    do_not_indent_me_either = 2\n'
#        '                    ifIam_indented_something_went_wrong = 2\n'
#        '                    if_I_am_indented_something_went_wrong = 2\n'
#        '                    end should not be unindented here\n'
#        '                end if\n'
#        '            endif\n'
#        '        end do\n'
#        '    end do\n'
#        'enddo\n'
#        'end subroutine\n'
#        '\n'
#        'subroutine test2(a)\n'
#        'integer :: a\n'
#        'do\n'
#        '    a = a + 1\n'
#        'end do \n'
#        'end subroutine\n'
#    )
#    p = FCodePrinter()
#    result = p.indent_code(codelines)
#    assert result == expected
#
#
#def test_lua_code_Assign():
#    assert lua_code(Assign(x, y + z)) == 'x = y + z'
#
#
#def test_lua_code_For():
#    f = For(x, Range(0, 10, 2), [Assign(y, x * y)])
#    sol = lua_code(f)
#    assert sol == ("do x = 0, 10, 2\n"
#                   "    y = x*y\n"
#                   "end do")
#
#
#def test_lua_code_Import():
#    assert lua_code(Import('math', 'sin')) == 'use math, only: sin'
#
#
#def test_lua_code_Declare():
#    assert lua_code(Declare('int', InArgument('int', a))) == "integer, intent(in) :: a"
#    assert lua_code(Declare('double', (InArgument('double', a), InArgument('double',
#            b), OutArgument('double', c),
#            InOutArgument('double', x)))) == ("real(dp), intent(in) :: a, b\n"
#                                              "real(dp), intent(inout) :: x\n"
#                                              "real(dp), intent(out) :: c")

################################################
if __name__ == "__main__":
    test_lua_code_Pow()
