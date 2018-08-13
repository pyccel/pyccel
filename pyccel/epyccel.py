# coding: utf-8

from pyccel.parser.syntax.headers import parse
from pyccel.parser.errors import Errors
from pyccel.parser.errors import PyccelError

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.codegen.utilities import execute_pyccel
from pyccel.ast import FunctionHeader

from collections import OrderedDict

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


def compile_fortran(source, modulename, extra_args='',libs = [], compiler = None):
    """use f2py to compile a source code. We ensure here that the f2py used is
    the right one with respect to the python/numpy version, which is not the
    case if we run directly the command line f2py ..."""
    
    if compiler:
        compiler = '--fcompiler={}'.format(compiler)
    else:
        compiler  = ''

    try:
        filename = '{}.f90'.format(modulename)
        f = open(filename, "w")
        for line in source:
            f.write(line)
        f.close()
        libs = ' '.join('-l'+i.lower() for i in libs)
        args = """ {} -c {}  --opt='-O3' -m  {} {} {}  """.format(compiler, libs, modulename, filename, extra_args)
        import sys
        cmd = 'f2py {}'.format(args)
        output = subprocess.check_output(cmd, shell=True)
        return output, cmd

    finally:
        f.close()


def epyccel(func, inputs, verbose=False, modules=[], libs=[], name=None,
            context=None, compiler = None):
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

    context: ContextPyccel, list/tuple
        a Pyccel context for user defined functions and other dependencies
        needed to compile func. it also be a list/tuple of ContextPyccel


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

    extra_args = ''

    if context:
        if isinstance(context, ContextPyccel):
            context = [context]
        elif isinstance(context, (list, tuple)):
            for i in context:
                assert(isinstance(i, ContextPyccel))
        else:
            raise TypeError('Expecting a ContextPyccel or list/tuple of ContextPyccel')

        imports = []
        names_o = []
        for i in context:
            names_o.append('{name}.o'.format(name=i.name))
            imports.append(i.imports)

        extra_args = ' '.join(i for i in names_o)
        imports = '\n'.join(i for i in imports)
        # ...

        # ... add import to initial code
        code = '{imports}\n{code}'.format(imports=imports, code=code)
        # ...

    try:
        # ...
        pyccel = Parser(code, headers={str(name): header})
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        codegen = Codegen(ast, name)
        code = codegen.doprint()
        # ...

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    except:
        # reset Errors singleton
        errors = Errors()
        errors.reset()

        raise PyccelError('Could not convert to Fortran')

    output, cmd = compile_fortran(code, name, extra_args=extra_args, libs = libs, compiler = compiler)

    if verbose:
        print(cmd)

  #  if verbose:
  #      print(output)
    # ...

    # ...
    try:
        package = importlib.import_module(name)
        # TODO ??
        #reload(package)
    except:
        raise ImportError('could not import {0}'.format(name))
    # ...
    if name in dir(package):
        module = getattr(package, name)
    else:
        module = getattr(package, 'mod_{0}'.format(name.lower()))
    
    #f = getattr(module, name.lower())

    return module


# TODO check what we are inserting
class ContextPyccel(object):
    """Class for interactive use of Pyccel. It can be used within an IPython
    session, Jupyter Notebook or ipyccel command line."""
    def __init__(self, name):
        self._name = 'mod_{}'.format(name)
        self._constants = OrderedDict()
        self._functions = OrderedDict()

    @property
    def name(self):
        return self._name

    @property
    def constants(self):
        return self._constants

    @property
    def functions(self):
        return self._functions

    def __str__(self):
        # line separator
        sep = '# ' + '.'*30 + '\n'

        # ... constants
        # TODO remove # if we want to export constants too
        constants = '\n'.join('#{k} = {v}'.format(k=k,v=v) for k,v in list(self.constants.items()))
        # ...

        # ... functions
        functions = ''
        for k, (f,h) in list(self.functions.items()):
            code_f = get_source_function(f)

            functions = '{h}\n{f}\n{functions}'.format(h=h, f=code_f, functions=functions)
        # ...

        code = '{sep}{constants}\n{sep}{functions}'.format(sep=sep,
                                                           constants=constants,
                                                           functions=functions)

        return code

    def insert_constant(self, d, value=None):
        """Inserts constants in the namespace.

        d: str, dict
            an identifier string or a dictionary of the form {'a': value_a, 'b': value_b} where `a` and
            `b` are the constants identifiers and value_a, value_b their
            associated values.

        value: int, float, complex, str
            value used if d is a string

        """
        if isinstance(d, str):
            if value is None:
                raise ValueError('Expecting a not None value')

            self._constants[d] = value

        elif isinstance(d, (dict, OrderedDict)):
            for k,v in list(d.items()):
                self._constants[k] = v

        else:
            raise ValueError('Expecting d to be a string or dict/OrderedDict')


    def insert_function(self, func, types, kind='function', results=None):
        """Inserts a function in the namespace."""
        # function name
        name = func.__name__

        # ... construct a header from d_types
        assert(isinstance(types, (list, tuple)))

        types = ', '.join('{}'.format(i) for i in types)

        header = '#$ header {kind} {name}'.format(name=name, kind=kind)
        header = '{header}({types})'.format(header=header, types=types)

        if results:
            results = ', '.join('{}'.format(i) for i in results)
            header = '{header} results({results})'.format(header=header, results=results)
        # ...

        self._functions[name] = (func, header)

    # TODO add other things to import apart from functions
    @property
    def imports(self):
        """Returns available imports from the context as a string."""
        f_names = list(self.functions.keys())
        f_names = ', '.join(f for f in f_names)

        import_stmt = 'from {mod} import {f_names}'.format(mod=self.name,
                                                          f_names=f_names)

        return import_stmt

    def compile(self, **settings):
        """Convert to Fortran and compile the context."""
        code = self.__str__()

        # ... export the python code of the module
        filename = '{}.py'.format(self.name)
        f = open(filename, 'w')
        for line in code:
            f.write(line)
        f.close()
        # ...

        # ...
        verbose = settings.pop('verbose', False)

        if not('fflags' in list(settings.keys())):
            settings['fflags'] = '-fPIC -O3'
        # ...

        output, cmd = execute_pyccel(filename, verbose=verbose, **settings)
        return output, cmd
