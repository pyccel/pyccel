# coding: utf-8

from importlib.machinery import ExtensionFileLoader
from collections         import OrderedDict
from types               import ModuleType, FunctionType

import inspect
import subprocess
import importlib
import sys
import os

from pyccel.parser                import Parser
from pyccel.parser.errors         import Errors, PyccelError
from pyccel.parser.syntax.headers import parse
from pyccel.codegen               import Codegen
from pyccel.codegen.utilities     import execute_pyccel
from pyccel.ast                   import FunctionHeader

#==============================================================================

PY37 = sys.version_info[0:2] == (3, 7) 

#==============================================================================

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

#==============================================================================

def compile_fortran(source, modulename, extra_args='',libs=[], compiler=None , mpi=False, includes = []):
    """use f2py to compile a source code. We ensure here that the f2py used is
    the right one with respect to the python/numpy version, which is not the
    case if we run directly the command line f2py ..."""

    compilers  = ''

    if mpi:
        compilers = '--f90exec=mpif90 '

    if compiler:
        compilers = compilers +'--fcompiler={}'.format(compiler)

    try:
        filename = '{}.f90'.format( modulename.replace('.','/') )
        filename = os.path.basename( filename )
        f = open(filename, "w")
        for line in source:
            f.write(line)
        f.close()
        libs = ' '.join('-l'+i.lower() for i in libs)
        args = """  -c {} --f90flags=-O3 {} -m  {} {} {} {} """.format(compilers,
                                                libs, modulename.rpartition('.')[2], filename,
                                                extra_args, includes)

        cmd = """python3 -m numpy.f2py {}"""


        cmd = cmd.format(args)
        output = subprocess.check_output(cmd, shell=True)
        return output, cmd

    finally:
        f.close()

#==============================================================================

def epyccel(func, inputs=None, verbose=False, modules=[], libs=[], name=None,
            context=None, compiler = None , mpi=False, static=None):
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

    static: list/tuple
        a list of 'static' functions as strings

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
    is_module = False
    is_function = False

    if isinstance(func, ModuleType):
        is_module = True

    if callable(func):
        is_function = True

    assert(callable(func) or isinstance(func, str) or isinstance(func, ModuleType))

    # ...
    if callable(func) or isinstance(func, ModuleType):
        name = func.__name__

    elif name is None:
        # case of func as a string
        raise ValueError('function name must be provided, in the case of func string')
    # ...

    output_folder = name.rsplit('.',1)[0] if '.' in name else ''

    # fortran module name
    modname = 'epyccel__' + name



    # ...
    if is_module:
        mod = func
        is_sharedlib = isinstance(getattr(mod, '__loader__', None), ExtensionFileLoader)

        if is_sharedlib:
            module_filename = inspect.getfile(mod)

            # clean
            cmd = 'rm -f {}'.format(module_filename)
            os.system(cmd)

            # then re-run again
            mod = importlib.import_module(name)
            # we must reload the module, otherwise it is still the .so one
            importlib.reload(mod)
            epyccel(mod, inputs=inputs, verbose=verbose, modules=modules,
                    libs=libs, name=name, context=context, compiler=compiler,
                    mpi=mpi, static=static)
    # ...

    # ...
    ignored_funcs = None
    if not static:
        if isinstance(func, ModuleType):
            mod = func
            funcs = [i for i in dir(mod) if isinstance(getattr(mod, i), FunctionType)]

            # remove pyccel.decorators
            ignored_funcs = [i for i in funcs if getattr(mod, i).__module__ == 'pyccel.decorators']
            static = [i for i in funcs if not(i in ignored_funcs)]

        else:
            static = [name]
    # ...

    # ...
    headers = None
    if inputs:
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
    d_headers = {}
    if headers:
        hdr = parse(stmts=headers)
        if isinstance(hdr, FunctionHeader):
            header = hdr.to_static()
            d_headers = {str(name): header}

        elif isinstance(hdr, (tuple, list)):
            hs = [h.to_static() for h in hdr]
            hs = [h for h in hs if hs.func == name]
            # TODO improve
            header = hs[0]
            raise NotImplementedError('TODO')

        else:
            raise NotImplementedError('TODO')
    # ...

    # ...
    if not static:
        raise NotImplementedError('TODO')
    # ...

    # ... get the function source code
    if callable(func):
        code = get_source_function(func)
        print(code)

    elif isinstance(func, ModuleType):
        lines = inspect.getsourcelines(func)[0]
        code = ''.join(lines)

    else:
        code = func
    # ...

    if verbose:
        print ('------')
        print (code)
        print ('------')

    extra_args = ''
    include_args = ''

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
        include_folders = []
        context_import_path = []
        for i in context:
            names_o.append('{fol}{name}.o'.format(fol=i.os_folder,name=i.name))
            include_folders.append('-I{}'.format(i.os_folder))
            imports.append(i.imports)
            context_import_path.append((i.name,i.os_folder))
        context_import_path = dict(context_import_path)

        extra_args = ' '.join(i for i in names_o)
        include_args = ' '.join(i for i in include_folders)
        imports = '\n'.join(i for i in imports)
        # ...

        # ... add import to initial code
        code = '{imports}\n{code}'.format(imports=imports, code=code)
        # ...
    else:
        context_import_path = {}

    try:
        # ...
        pyccel = Parser(code, headers=d_headers, static=static, output_folder = output_folder,
                            context_import_path=context_import_path)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        codegen = Codegen(ast, modname)
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

       # Change module name to avoid name clashes: Python cannot import two modules with the same name
    if is_module:
        head, sep, tail = name.rpartition('.')
        name = sep.join( [head, '__epyccel__'+ tail] )
    else:
        name = '__epyccel__'+ name


    # Find directory where Fortran extension module should be created
    if is_module:
        dirname = os.path.dirname(os.path.abspath( mod.__file__ ))
    else:
        dirname = os.path.dirname(os.path.abspath(sys.modules[func.__module__].__file__ ))

    # Move into working directory, create extension module, then move back to original directory
    origin = os.path.abspath( os.curdir )
    os.chdir( dirname )
    output, cmd = compile_fortran( code, name,
        extra_args= extra_args,
        libs      = libs,
        compiler  = compiler,
        mpi       = mpi,
        includes  = include_args)
    os.chdir( origin )

    if verbose:
        print(cmd)

  #  if verbose:
  #      print(output)
    # ...

    # ...
    try:
        if PY37:
            dirname = os.path.relpath(dirname).replace('/','.')
            package = dirname + '.' + name
            package = importlib.import_module( '..'+name, package=package )
            clean_extension_module( package, modname )
        else:
            os.chdir( dirname )
            package = importlib.import_module( name )
            clean_extension_module( package, modname )
            os.chdir( origin )

    except:
        raise ImportError('could not import {0}'.format( name ))
    # ...

    if is_module:
        return package
    else:
        return getattr( package, func.__name__.lower() )

#==============================================================================

# TODO: write similar version for single functions
def epyccel_mpi( mod, comm, root=0 ):
    """
    Collective version of epyccel for modules: root process generates Fortran
    code, compiles it and creates a shared library (extension module), which
    is then loaded by all processes in the communicator.

    Parameters
    ----------
    mod : types.ModuleType
        Python module to be pyccelized.

    comm: mpi4py.MPI.Comm
        MPI communicator where extension module will be made available.

    root: int
        Rank of process responsible for code generation.

    Results
    -------
    fmod : types.ModuleType
        Python extension module.

    """
    from mpi4py import MPI

    assert isinstance(  mod, ModuleType )
    assert isinstance( comm, MPI.Comm   )
    assert isinstance( root, int        )

    # Master process calls epyccel
    if comm.rank == root:
        fmod      = epyccel( mod, mpi=True )
        fmod_path = fmod.__file__
        fmod_name = fmod.__name__
    else:
        fmod_path = None
        fmod_name = None

    # Broadcast Fortran module path/name to all processes
    fmod_path = comm.bcast( fmod_path, root=root )
    fmod_name = comm.bcast( fmod_name, root=root )

    # Non-master processes import Fortran module directly from its path
    if comm.rank != root:
        spec = importlib.util.spec_from_file_location( fmod_name, fmod_path )
        fmod = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( fmod )
        clean_extension_module( fmod, mod.__name__ )

    # Return Fortran module
    return fmod

#==============================================================================

def clean_extension_module( ext_mod, py_mod_name ):
    """
    Clean Python extension module by moving functions contained in f2py's
    "mod_[py_mod_name]" automatic attribute to one level up (module level).
    "mod_[py_mod_name]" attribute is then completely removed from the module.

    Parameters
    ----------
    ext_mod : types.ModuleType
        Python extension module created by f2py from pyccel-generated Fortran.

    py_mod_name : str
        Name of the original (pure Python) module.

    """
    # Get name of f2py automatic attribute
    n = py_mod_name.lower().replace('.','_')

    # Move all functions to module level
    m = getattr( ext_mod, n )
    for a in type( m ).__dir__( m ):
        if a.startswith( '__' ) and a.endswith( '__' ):
            pass
        else:
            setattr( ext_mod, a, getattr( m, a ) )

    # Remove f2py automatic attribute
    delattr( ext_mod, n )

#==============================================================================

# TODO check what we are inserting
class ContextPyccel(object):
    """Class for interactive use of Pyccel. It can be used within an IPython
    session, Jupyter Notebook or ipyccel command line."""
    def __init__(self, name, context_folder=None, output_folder=''):
        self._name = 'epyccel__{}'.format(name)
        self._constants = OrderedDict()
        self._functions = OrderedDict()

        if not context_folder:
            context_folder = os.path.abspath('.').replace('/', '.')

        self._folder = context_folder
        if (len(self._folder)>0):
            self._folder+='.'
        self._os_folder = self._folder.replace('.','/')
        
        contexts = context_folder.split('.')
        outputs  =  output_folder.split('.')
        n = min(len(contexts),len(outputs))
        i=0
        while(i<n and contexts[i]==outputs[i]):
            i+=1
        contexts = contexts[i:]
        outputs  =  outputs[i:]

        if (len(contexts)==0 and len(outputs)==0):
            self._rel_folder = ''
        else:
            self._rel_folder = '.'*(len(outputs)+1)+'.'.join(contexts)
            if (self._rel_folder[-1]!='.'):
                self._rel_folder+='.'

    @property
    def folder(self):
        return self._folder

    @property
    def os_folder(self):
        return self._os_folder

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

        output, cmd = execute_pyccel(filename, verbose=verbose, **settings)
        return output, cmd
