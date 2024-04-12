"""
File describing commands associated with the lambdify function which converts a
SymPy expression into a Pyccel-accelerated function.
"""
import sympy as sp
from packaging.version import parse

from pyccel.epyccel import epyccel
from pyccel.utilities.strings import random_string

if parse(sp.__version__) >= parse('1.8'):
    from sympy.printing.numpy import NumPyPrinter
else:
    from sympy.printing.pycode import NumPyPrinter

def lambdify(expr : sp.Expr, args : 'dict[sp.Symbol, str]', result_type : str = None,
             templates : 'dict[str,list[str]]' = None, **kwargs):
    """
    Convert a SymPy expression into a Pyccel-accelerated function.

    Convert a SymPy expression into a function that allows for fast
    numeric evaluation. This is done using sympy's NumPyPrinter to
    generate code that can be accelerated by Pyccel.

    Parameters
    ----------
    expr : sp.Expr
        The sympy expression that should be returned from the function.
    args : dict[sp.Symbol, str]
        A dictionary of the arguments of the function being created.
        The keys are variables representing the arguments that will be
        passed to the function. The values are the the type annotations
        for those functions.
    result_type : str, optional
        The type annotation for the result of the function. This argument
        is optional but it is recommended to provide it as sympy
        expressions do not always evaluate to the expected type. For
        example if the sympy expression simplifies to 0 then the default
        type will be int even if the arguments are floats.
    templates : dict[str,list[str]], optional
        A description of any templates that should be added to the
        function. The keys are the symbols which can be used as type
        specifiers, the values are a list of the type annotations which
        are valid types for the symbol. See
        <https://github.com/pyccel/pyccel/blob/devel/docs/templates.md>
        for more details.
    **kwargs : dict
        Additional arguments that are passed to epyccel.

    Returns
    -------
    func
        A Pyccel-accelerated function which allows the evaluation of
        the sympy expression.

    See Also
    --------
    sympy.lambdify
        <https://docs.sympy.org/latest/modules/utilities/lambdify.html>
    epyccel
        The function that accelerates the generated code.
    """
    expr = NumPyPrinter().doprint(expr)
    args = ', '.join(f'{a} : "{annot}"' for a, annot in args.items())
    func_name = 'func_'+random_string(8)
    if result_type:
        signature = f'def {func_name}({args}) -> "{result_type}":'
    else:
        signature = f'def {func_name}({args}):'
    if templates:
        decorators = '\n'.join(f'@template({key}, ['+', '.join(f'"{annot}"' for annot in annotations)+']' \
                for key, annotations in templates.items())
    else:
        decorators = ''
    code = f'    return {expr}'
    numpy_import = 'import numpy\n'
    func = '\n'.join((numpy_import, decorators, signature, code))
    print(func)
    package = epyccel(func, **kwargs)
    return getattr(package, func_name)

