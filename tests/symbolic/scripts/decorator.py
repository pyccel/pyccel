# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

#@sympy
#def g_sympy(t,i0):
#    #This function must return a sympy expression
#    #that depends on the arguments of the function
#    from sympy import symbols, Eq, dsolve, solve, Derivative, sin
#    x = sympify
#    t = symbols('t')
#    i0 = symbols('i0')
#    eq = Eq(Derivative(x(t),t),x(t)*sin(t))
#    expr = dsolve(eq).args[1]
#    sol = solve(expr.subs(t,0)-i0)
#    expr = expr.subs(sol[0].items())
#    return expr
#
##$ header function  g(double, double)


@sympy
def f1_sympy():
    #This function must return a sympy expression
    #that depends on the arguments of the function
    from sympy.abc import x,y
    expr = x*y + 2
    return expr

@sympy
def f2_sympy(x):
    #This function must return a sympy expression
    #that depends on the arguments of the function
    from sympy import diff
    u = x**2 + 2*x
    u_dx = diff(u, x)
    return u_dx
