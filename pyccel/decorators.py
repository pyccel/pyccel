#TODO improve sympy to handle Sum objects

def sympy(f):
    args = f.__code__.co_varnames
    from sympy import symbols
    args = symbols(args)
    expr = f(*args)
    def wrapper(*vals):
       return  expr.subs(zip(args,vals)).doit()
    
    return wrapper


def python(f):
    args = f.__code__.co_varnames
    def wrapper(*vals):
       return  f(*vals)

def lambdify(f):
    return f


def types(f):
    return f



