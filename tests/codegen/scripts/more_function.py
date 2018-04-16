from numpy import cos
from numpy import exp
#$header function f(double)
#$header function g(double,double)
@python
def g(t,i0):
    from sympy import symbols, Eq, dsolve, solve, Derivative, sin
    x = symbols('x')
    t = symbols('t')
    i0 = symbols('i0')
    eq = Eq(Derivative(x(t),t),x(t)*sin(t))
    expr = dsolve(eq).args[1]
    sol = solve(expr.subs(t,0)-i0)
    expr = expr.subs(sol[0].items())
    return expr

def f(x):
    return x**2,x+1,g(x-1,5.)




