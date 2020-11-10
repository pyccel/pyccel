# coding: utf-8

from sympy import Symbol, sympify, Tuple
from sympy import Poly, LT
from sympy.core.expr import Expr


from pyccel.ast.core     import For, Assign, NewLine, CodeBlock, Comment
from pyccel.ast.numpyext import NumpyZeros, NumpyOnes
from pyccel.complexity.basic import Complexity


__all__ = ["count_access", "MemComplexity"]

# ...
def count_access(expr, visual=True):
    """
    returns the number of access to memory in terms of WRITE and READ.

    expr: sympy.Expr
        any sympy expression or pyccel.ast.core object
    visual: bool
        If ``visual`` is ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.
    local_vars: list
        list of variables that are supposed to be in the fast memory. We will
        ignore their corresponding memory accesses.
    """

    WRITE = Symbol('WRITE')
    READ  = Symbol('READ')


    if isinstance(expr, Expr):

        atoms = expr.atoms(Symbol)
        return READ*len(atoms)

    elif isinstance(expr, Assign):
        return count_access(expr.rhs, visual) + WRITE

    elif isinstance(expr, Tuple):
        return sum(count_access(i, visual) for i in expr)

    elif isinstance(expr, CodeBlock):
        return sum(count_access(i, visual) for i in expr.body)

    elif isinstance(expr, For):
        s = expr.iterable.size
        ops = sum(count_access(i, visual) for i in expr.body.body)
        return ops*s

    elif isinstance(expr, (NumpyZeros, NumpyOnes)):
        import numpy as np
        return WRITE*np.prod(expr.shape)

    elif isinstance(expr, (NewLine, Comment)):
        return 0
    else:
        raise NotImplementedError('TODO count_access for {}'.format(type(expr)))



def leading_term(expr, *args):
    """
    Returns the leading term in a sympy Polynomial.

    expr: sympy.Expr
        any sympy expression

    args: list
        list of input symbols for univariate/multivariate polynomials
    """
    expr = sympify(str(expr))
    P = Poly(expr, *args)
    return LT(P)
# ...

# ...
class MemComplexity(Complexity):
    """
    Class for memory complexity computation.
    This class implements a simple two level memory model

    Example

    >>> code = '''
    ... n = 10
    ... for i in range(0,n):
    ...     for j in range(0,n):
    ...         x = pow(i,2) + pow(i,3) + 3*i
    ...         y = x / 3 + 2* x
    ... '''

    >>> from pyccel.complexity.memory import MemComplexity
    >>> M = MemComplexity(code)
    >>> d = M.cost()
    >>> print "f = ", d['f']
    f =  n**2*(2*ADD + DIV + 2*MUL + 2*POW)
    >>> print "m = ", d['m']
    m =  WRITE + 2*n**2*(READ + WRITE)
    >>> q = M.intensity()
    >>> print "+++ computational intensity ~", q
    +++ computational intensity ~ (2*ADD + DIV + 2*MUL + 2*POW)/(2*READ + 2*WRITE)

    Now let us consider a case where some variables are supposed to be in the
    fast memory, (*r* in this test)

    >>> code = '''
    ... n = 10
    ... x = zeros(shape=(n,n), dtype=float)
    ... r = float()
    ... r = 0
    ... for i in range(0, n):
    ...     r = x[n,i] + 1
    ... '''

    >>> M = MemComplexity(code)
    >>> d = M.cost()
    >>> print "f = ", d['f']
    f =  ADD*n
    >>> print "m = ", d['m']
    m =  2*WRITE + n*(READ + WRITE)
    >>> q = M.intensity()
    >>> print "+++ computational intensity ~", q
    +++ computational intensity ~ ADD/(READ + WRITE)

    Notice, that this is not what we expect! the cost of writing into *r* is
    'zero', and therefor, there should be no :math:`n*WRITE` in our memory cost.
    In order to achieve this, you must tell pyccel that you have the variable
    *r* is already in the fast memory. This can be done by adding the argument
    *local_vars=['r']* when calling the cost method.

    >>> d = M.cost(local_vars=['r'])
    >>> print "f = ", d['f']
    f =  ADD*n
    >>> print "m = ", d['m']
    m =  READ*n + WRITE
    >>> q = M.intensity(local_vars=['r'])
    >>> print "+++ computational intensity ~", q
    +++ computational intensity ~ ADD/READ

    and this is exactly what we were expecting.
    """
    def cost(self):
        """
        Computes the complexity of the given code.

        local_vars: list
            list of variables that are supposed to be in the fast memory. We will
            ignore their corresponding memory accesses.
        """

        return count_access(self.ast, visual=True)

    def intensity(self, d=None, args=None, local_vars=[], verbose=False):
        """
        Returns the computational intensity for the two level memory model.

        d: dict
            dictionary containing the floating and memory costs. if not given,
            we will compute them.
        args: list
            list of free parameters, i.e. degrees of freedom.
        local_vars: list
            list of variables that are supposed to be in the fast memory. We will
            ignore their corresponding memory accesses.
        verbose: bool
            talk more
        """
        # ...
        if d is None:
            d = self.cost(local_vars=local_vars)
        # ...

        # ...
        if args is None:
            args = self.free_parameters
        # ...

        # ...
        f = d['f']
        m = d['m']
        # ...

        # ...
        lt_f = leading_term(f, *args)
        lt_m = leading_term(m, *args)

        q = lt_f/lt_m
        # ...

        # ...
        if verbose:
            print((" arithmetic cost         ~ " + str(f)))
            print((" memory cost             ~ " + str(m)))
            print((" computational intensity ~ " + str(q)))
        # ...

        return q


