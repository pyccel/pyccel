#TODO use pycode and call exec after that in lambdify

def lambdify(f):

    args = f.__code__.co_varnames
    from sympy import symbols
    args = symbols(args)
    expr = f(*args)
    def wrapper(*vals):
       return  expr.subs(zip(args,vals)).doit()

    return wrapper


def python(f):
    return f

def sympy(f):
    return f

def bypass(f):
    return f

def types(*args,**kw):
    def id(f):
        return f
    return id


def pure(f):
    return f

def private(f):
    return f

def elemental(f):
    return f

def external(f):
    return f

def external_call(f):
    return f

def stack_array(*args, **kw):
    def id(f):
        return f
    return id

# TODO documentation
import numpy as np
class f2py_compatible(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        newargs = []
        for a in args:
            if ( isinstance(a, np.ndarray) and
                ( len(a.shape) > 1)        and
                a.flags['C_CONTIGUOUS'] ):
                newargs.append(a.transpose())

            else:
                newargs.append(a)

        return self.f(*newargs)
