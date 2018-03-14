
from pyccel.parser.syntax.headers import parse
from pyccel.parser import Parser
from pyccel.codegen import Codegen

import inspect
import subprocess
import importlib


def epyccel(func, inputs, verbose=False, modules=[], libs=[]):
    """Pyccelize a python function and wrap it using f2py.

    func: function, str
        a Python function or source code defining the function

    inputs: str, list, tuple, dict
        inputs can be the function header as a string, or a list/tuple of
        strings or the globals() dictionary

    verbose: bool
        talk more

    modules: list, tuple
        list of dependencies

    libs: list, tuple
        list of libraries


    Examples

    The following example shows how to use Pyccel within an IPython session

    >>> #$ header procedure static f_static(int [:]) results(int)
    >>> def f_static(x):
    >>>     y = x[0] - 1
    >>>     return y

    >>> from test_epyccel import epyccel
    >>> f = epyccel(f_static, globals()) # appending IPython history

    >>> header = '#$ header procedure static f_static(int [:]) results(int)'
    >>> f = epyccel(f_static, header) # giving the header explicitly

    Now, **f** is a Fortran function that has been wrapped. It is compatible
    with numpy and you can call it

    >>> import numpy as np
    >>> x = np.array([3, 4, 5, 6], dtype=int)
    >>> y = f(x)

    You can also call it with a list instead of numpy arrays

    >>> f([3, 4, 5])
    2
    """
    assert(callable(func) or isinstance(func, str))

    if isinstance(func, str):
        raise NotImplementedError('Treat the case of string source code')

    name = func.__name__

    # ...
    if isinstance(inputs, str):
        headers = inputs
    elif isinstance(inputs, (tuple, list)):
        # find all possible headers
        lines = [str(i) for i in inputs if (isinstance(i, str) and
                                            i.lstrip().startswith('#$ header'))]
        # TODO take the last occurence for f => use zip
        headers = "\n".join([str(i) for i in lines])
    elif isinstance(inputs, dict):
        # case of globals() history from ipython
        if not 'In' in inputs.keys():
            raise ValueError('Expecting `In` key in the inputs dictionary')

        inputs = inputs['In']

        # TODO shall we reverse the list

        # find all possible headers
        lines = [str(i) for i in inputs if i.lstrip().startswith('#$ header')]
        # TODO take the last occurence for f => use zip
        headers = "\n".join([str(i) for i in lines])
    # ...

    # ...
    # get the function source code
    lines = inspect.getsourcelines(func)
    func_str = "".join(lines[0])

    code = '{headers}\n{func}'.format(headers=headers, func=func_str)
    # ...

    if verbose:
        print ('------')
        print code
        print ('------')

    # ...
    pyccel = Parser(code)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    codegen = Codegen(ast, name)
    code = codegen.doprint()
    codegen.export()
    # ...

    # ...
    filename = '{name}.f90'.format(name=name)
    # ...

    # ...
    def _format_libs(ls):
        if not ls:
            ls = ''
        elif isinstance(ls, (tuple, list)):
            ls = ' '.join(i for i in ls)
        elif not isinstance(ls, str):
            raise TypeError('Expecting a list or a string for libs')
        return ls

    def _format_modules(ls):
        if isinstance(ls, (tuple, list)):
            ls = ''.join('{}.o '.format(m) for m in ls)
        else:
            raise TypeError('Expecting a list/tuple')
        return ls
    # ...

    # TODO improve
    flags  = '--quiet -c'

    binary = name
    compiler = 'f2py'
    libs = _format_libs(libs)
    deps_o = _format_modules(modules)

    cmd = 'f2py {flags} {deps_o} {filename} -m {binary} {libs}'.format(flags=flags,
                                                                       filename=filename,
                                                                       binary=binary,
                                                                       libs=libs,
                                                                       deps_o=deps_o)

    if verbose:
        print(cmd)

    output = subprocess.check_output(cmd, shell=True)

    if verbose:
        print(output)
    # ...

    # ...
    try:
        package = importlib.import_module(name)
        # TODO ??
        #reload(package)
    except:
        raise ImportError('could not import {0}'.format(name))
    # ...

    module = getattr(package, 'mod_{0}'.format(name))
    f = getattr(module, name)

    return f
