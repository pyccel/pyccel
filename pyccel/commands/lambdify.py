"""
File describing commands associated with the lambdify function which converts a
SymPy expression into a Pyccel-accelerated function.
"""
import sympy as sp
from packaging import version

from pyccel.commands.epyccel import epyccel
from pyccel.utilities.strings import random_string
from pyccel.errors.errors     import PyccelError

if version.parse(sp.__version__) >= version.parse('1.8'):
    from sympy.printing.numpy import NumPyPrinter
else:
    from sympy.printing.pycode import NumPyPrinter

def lambdify(expr : sp.Expr, args : 'dict[sp.Symbol, str]', *, result_type : str = None,
             templates : 'dict[str, list[str]]' = None, use_out = False,
             **kwargs):
    """
    Convert a SymPy expression into a Pyccel-accelerated function.

    Convert a SymPy expression into a function that allows for fast
    numeric evaluation. This is done using SymPy's NumPyPrinter to
    generate code that can be accelerated by Pyccel.

    Parameters
    ----------
    expr : sp.Expr
        The SymPy expression that should be returned from the function.
    args : dict[sp.Symbol, str]
        A dictionary of the arguments of the function being created.
        The keys are variables representing the arguments that will be
        passed to the function. The values are the the type annotations
        for those functions.
    result_type : str, optional
        The type annotation for the result of the function. This argument
        is optional but it is recommended to provide it as SymPy
        expressions do not always evaluate to the expected type. For
        example if the SymPy expression simplifies to 0 then the default
        type will be int even if the arguments are floats.
    templates : dict[str, list[str]], optional
        A description of any templates that should be added to the
        function. The keys are the symbols which can be used as type
        specifiers, the values are a list of the type annotations which
        are valid types for the symbol. See
        <https://github.com/pyccel/pyccel/blob/devel/docs/templates.md>
        for more details.
    use_out : bool, default=False
        If true the function will modify an argument called 'out' instead
        of returning a newly allocated array. If this argument is set then
        result_type must be provided. This only works if the result is an
        array type.
    **kwargs : dict
        Additional arguments that are passed to epyccel.

    Returns
    -------
    func
        A Pyccel-accelerated function which allows the evaluation of
        the SymPy expression.

    See Also
    --------
    sympy.lambdify
        <https://docs.sympy.org/latest/modules/utilities/lambdify.html>.
    epyccel
        The function that accelerates the generated code.
    """
    if not (isinstance(args, dict) and all(isinstance(k, sp.Symbol) and isinstance(v, str) for k,v in args.items())):
        raise TypeError("Argument 'args': Expected a dictionary mapping SymPy symbols to string type annotations.")
    if result_type is not None and not isinstance(result_type, str):
        raise TypeError("Argument 'result_type': Expected a string type annotation.")

    expr = NumPyPrinter().doprint(expr)
    args_code = ', '.join(f'{a} : "{annot}"' for a, annot in args.items())
    func_name = 'func_'+random_string(8)

    docstring = " \n".join(('    """',
            "    Expression evaluation created with `pyccel.lambdify`.",
            "",
            "    Function evaluating the expression:",
           f"    {expr}",
            "",
            "    Parameters",
            "    ----------\n"))
    docstring += '\n'.join(f"    {a} : {type_annot}" for a, type_annot in args.items())

    if use_out:
        if not result_type:
            raise TypeError("The result_type must be provided if use_out is true.")
        else:
            signature = f'def {func_name}({args_code}, out : "{result_type}"):'
            docstring += f"\n    out : {result_type}"
    elif result_type:
        signature = f'def {func_name}({args_code}) -> "{result_type}":'
        docstring += "\n".join(("\n",
                    "     Returns",
                    "     -------",
                   f"     {result_type}"))
    else:
        signature = f'def {func_name}({args_code}):'
    if templates:
        if not (isinstance(templates, dict) and all(isinstance(k, str) and hasattr(v, '__iter__') for k,v in templates.items()) \
                and all(all(isinstance(type_annot, str) for type_annot in v) for v in templates.values())):
            raise TypeError("Argument 'templates': Expected a dictionary mapping strings describing type specifiers to lists of string type annotations.")

        decorators = '\n'.join(f'@template("{key}", ['+', '.join(f'"{annot}"' for annot in annotations)+'])' \
                for key, annotations in templates.items())
    else:
        decorators = ''
    if use_out:
        code = f'    out[:] = {expr}'
    else:
        code = f'    return {expr}'
    numpy_import = 'import numpy\n'

    docstring += '\n    """'

    func = '\n'.join((numpy_import, decorators, signature, docstring, code))
    try:
        package = epyccel(func, **kwargs)
    except PyccelError as e:
        raise type(e)(str(e)) from None

    return getattr(package, func_name)

