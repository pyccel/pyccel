def sympy(f):
    args = f.__code__.co_varnames
    from sympy import symbols
    args = symbols(args)
    expr = f(*args)
    def wrapper(*vals):
       return  expr.subs(zip(args,vals))
    
    return wrapper


