
from pyccel.parser.syntax.headers import parse
from pyccel.parser.errors import Errors
from pyccel.parser.errors import PyccelError

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.codegen.utilities import execute_pyccel
from pyccel.ast import FunctionHeader

import inspect
import subprocess
import importlib
import sys

def get_source_function(func):
    if not callable(func):
        raise TypeError('Expecting a callable function')

    lines = inspect.getsourcelines(func)
    lines = lines[0]
    # remove indentation if the first line is indented
    a = lines[0]
    leading_spaces = len(a) - len(a.lstrip())
    code = ''
    for a in lines:
        if leading_spaces > 0:
            line = a[leading_spaces:]
        else:
            line = a
        code = '{code}{line}'.format(code=code, line=line)

    return code


def compile_fortran(source, modulename, extra_args=''):
    """use f2py to compile a source code. We ensure here that the f2py used is
    the right one with respect to the python/numpy version, which is not the
    case if we run directly the command line f2py ..."""

    try:
        filename = '{}.f90'.format(modulename)
        f = open(filename, "w")
        for line in source:
            f.write(line)
        f.close()

        args = ' -c -m {} {} {}'.format(modulename, filename, extra_args)
        import sys
        cmd = '{} -c "import numpy.f2py as f2py2e;f2py2e.main()" {}'.format(sys.executable, args)

        output = subprocess.check_output(cmd, shell=True)
        return output, cmd

    finally:
        f.close()


def epyccel(func, inputs, verbose=False, modules=[], libs=[], name=None,
            d_functions={}):
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

    name: str
        name of the function, if it is given as a string

    d_functions: dict
        dictionary of used functions, the key is the function name, and the
        value is a tuple (func, header) where func is a python function.


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

    # ...
    if callable(func):
        name = func.__name__
    elif name is None:
        # case of func as a string
        raise ValueError('function name must be provided, in the case of func string')
    # ...

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

    # we parse all headers then convert them to static function
    hdr = parse(stmts=headers)
    if isinstance(hdr, FunctionHeader):
        header = hdr.to_static()
    elif isinstance(hdr, (tuple, list)):
        hs = [h.to_static() for h in hdr]
        hs = [h for h in hs if hs.func == name]
        # TODO improve
        header = hs[0]
    else:
        raise NotImplementedError('TODO')
    # ...

    # ... get the function source code
    if callable(func):
        code = get_source_function(func)
    else:
        code = func
    # ...

    if verbose:
        print ('------')
        print (code)
        print ('------')

    # ...
    code_module = ''
    if d_functions:

        sep = '-'*40
        sep = '# {sep}'.format(sep=sep)

        for k, (f,h) in list(d_functions.items()):
            code_f = get_source_function(f)

            code_module = '{h}\n{f}\n{code}'.format(h=h,
                                                    f=code_f,
                                                    code=code_module)
    # ...

    modules = []
    extra_args = ''

    # TODO add dependencies other than functions
    if d_functions:
        # ... we create a module for dependencies
        #module_name = 'mod_dep_'+ name
        module_name = 'mod_pyccel'
        dep_filename = module_name + '.py'
        # ...

        # ... export the python code of the module
        f = open(dep_filename, 'w')
        for line in code_module:
            f.write(line)
        f.close()
        # ...

        # ... TODO flags
        settings = {}
        settings['fflags'] = '-fPIC -O2'
        output, cmd = execute_pyccel(dep_filename, verbose=False, **settings)

        modules.append(module_name)
        extra_args = ' mod_pyccel.o '
        # ...

        # ... add import to initial code
        f_names = list(d_functions.keys())
        f_names = ', '.join(f for f in f_names)

        import_stmt = 'from {mod} import {f_names}'.format(mod=module_name,
                                                          f_names=f_names)
        code = '{IMPORT}\n{code}'.format(IMPORT=import_stmt, code=code)
        # ...

    # ...

    try:
        # ...
        pyccel = Parser(code, headers={str(name): header})
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        codegen = Codegen(ast, name)
        code = codegen.doprint()
#        codegen.export()
        # ...

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    except:
        # reset Errors singleton
        errors = Errors()
        errors.reset()

        raise PyccelError('Could not convert to Fortran')

    output, cmd = compile_fortran(code, name, extra_args=extra_args)

    if verbose:
        print(cmd)

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
