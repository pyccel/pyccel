# -*- coding: UTF-8 -*-

import os
from types import FunctionType

from pyccel.codegen.utilities       import construct_flags as construct_flags_pyccel
from pyccel.codegen.utilities       import execute_pyccel
from pyccel.epyccel import get_source_function
from pyccel.epyccel import random_string
from pyccel.epyccel import write_code
from pyccel.epyccel import mkdir_p
from pyccel.epyccel import get_function_from_ast

_avail_patterns = ['map']


#==============================================================================
def lambdify(pattern, *args, **kwargs):

    if not isinstance(pattern, str):
        raise TypeError('Expecting a string for pattern')

    if not pattern in _avail_patterns:
        raise ValueError('No pattern {} found'.format(pattern))

    _lambdify = eval('_lambdify_{}'.format(pattern))

    return _lambdify(*args, **kwargs)

#==============================================================================
def _lambdify_map(*args, **kwargs):

    # ... get arguments
    func, types = _extract_args_map(*args)
    # ...

    # ... get optional arguments
    namespace         = kwargs.pop('namespace'        , globals())
    compiler          = kwargs.pop('compiler'         , 'gfortran')
    fflags            = kwargs.pop('fflags'           , None)
    accelerator       = kwargs.pop('accelerator'      , None)
    verbose           = kwargs.pop('verbose'          , False)
    debug             = kwargs.pop('debug'            , False)
    include           = kwargs.pop('include'          , [])
    libdir            = kwargs.pop('libdir'           , [])
    modules           = kwargs.pop('modules'          , [])
    libs              = kwargs.pop('libs'             , [])
    extra_args        = kwargs.pop('extra_args'       , '')
    folder            = kwargs.pop('folder'           , None)
    mpi               = kwargs.pop('mpi'              , False)
    assert_contiguous = kwargs.pop('assert_contiguous', False)

    if fflags is None:
        fflags = construct_flags_pyccel( compiler,
                                         fflags=None,
                                         debug=debug,
                                         accelerator=accelerator,
                                         include=[],
                                         libdir=[] )
    # ...

    # ... get the function source code
    code = get_source_function(func)
    # ...

    # ...
    tag = random_string( 6 )
    # ...

    # ...
    module_name = 'mod_{}'.format(tag)
    fname       = '{}.py'.format(module_name)
    binary      = '{}.o'.format(module_name)
    # ...

    # ...
    if folder is None:
        basedir = os.getcwd()
        folder = '__pycache__'
        folder = os.path.join( basedir, folder )

    folder = os.path.abspath( folder )
    mkdir_p(folder)
    # ...

    # ...
    write_code(fname, code, folder=folder)
    # ...

    # ...
    basedir = os.getcwd()
    os.chdir(folder)
    curdir = os.getcwd()
    # ...

    # ...
    fname, ast = execute_pyccel( fname,
                                 compiler     = compiler,
                                 fflags       = fflags,
                                 debug        = debug,
                                 verbose      = verbose,
                                 accelerator  = accelerator,
                                 modules      = modules,
                                 convert_only = True,
                                 return_ast   = True )
    # ...

    # ... construct a f2py interface for the assembly
    # be careful: because of f2py we must use lower case
    func_name = func.__name__
    funcs     = ast.routines + ast.interfaces
    func      = get_function_from_ast(funcs, func_name)
    namespace = ast.parser.namespace.sons_scopes
    # ...

    print(func)
    print(func.decorators)

#==============================================================================
def _extract_args_map(*args):
    if not(len(args) >= 2):
        raise ValueError('Expecting at least 2 arguments')

    func = args[0]
    # TODO other args
    types = None

    if not isinstance(func, FunctionType):
        raise TypeError('> Expecting a function')

    return func, types
