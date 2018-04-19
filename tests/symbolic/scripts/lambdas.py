# coding: utf-8

#$ header function  g(double, double)
@sympy
def g(t,i0):
    #This function must return a sympy expression 
    #that depends on the arguments of the function
    from sympy import symbols, Eq, dsolve, solve, Derivative, sin
    x = sympify
    t = symbols('t')
    i0 = symbols('i0')
    eq = Eq(Derivative(x(t),t),x(t)*sin(t))
    expr = dsolve(eq).args[1]
    sol = solve(expr.subs(t,0)-i0)
    expr = expr.subs(sol[0].items())
    return expr


f1 = lambda x: x**2 + 1
f2 = lambda x,y: x**2 + f1(y)*f1(x)+g(x,y)
g1 = lambda x: f1(x)**2 + 1
