# coding: utf-8

"""
This file contains some useful functions to compile the generated fortran code
"""

import os
import subprocess

from pyccel.parser import Parser
from pyccel.codegen import Codegen

# TODO use constructor and a dict to map flags w.r.t the compiler
_avail_compilers = ['gfortran', 'mpif90', 'pgfortran']

# TODO add opt flags, etc... look at f2py interface in numpy
def construct_flags(compiler,
                    debug=False,
                    accelerator=None,
                    include=[],
                    libdir=[]):
    """
    Constructs compiling flags for a given compiler.

    compiler: str
        used compiler for the target language.

    accelerator: str
        name of the selected accelerator.
        One among ('openmp', 'openacc')

    debug: bool
        add some useful prints that may help for debugging.

    include: list
        list of include directories paths

    libdir: list
        list of lib directories paths
    """

    if not(compiler in _avail_compilers):
        raise ValueError("Only {0} are available.".format(_avail_compilers))

    flags = " -O2 "
    if compiler == "gfortran":
        if debug:
            flags += " -fbounds-check "

    if compiler == "mpif90":
        if debug:
            flags += " -fbounds-check "

    if not (accelerator is None):
        if accelerator == "openmp":
            flags += " -fopenmp "
        elif accelerator == "openacc":
            flags += " -ta=multicore -Minfo=accel "
        else:
            raise ValueError("Only openmp and openacc are available")

    if isinstance(include, str):
        include = [include]
    if len(include) > 0:
        flags += ' '.join(' -I{0}'.format(i) for i in include)

    if isinstance(libdir, str):
        libdir = [libdir]
    if len(libdir) > 0:
        flags += ' '.join(' -L{0}'.format(i) for i in libdir)

    return flags

def compile_fortran(filename, compiler, flags,
                    binary=None,
                    verbose=False,
                    modules=[],
                    is_module=False,
                    libs=[]):
    """
    Compiles the generated file.

    verbose: bool
        talk more
    """

    if binary is None:
        if not is_module:
            binary = os.path.splitext(os.path.basename(filename))[0]

    o_code = ''
    if not is_module:
        o_code = "-o"
    else:
        flags += ' -c '

    m_code = ' '.join('{}.o '.format(m) for m in modules)

    if isinstance(libs, str):
        libs = libs.split(',')
        if len(libs) == 1:
            libs = libs[0].split(' ')
    if len(libs) > 0:
        libs = ' '.join(' -l{0}'.format(i) for i in libs)
    else:
        libs = ''

    cmd = '{0} {1} {2} {3} {4} {5} {6}'.format( \
        compiler, flags, m_code, filename, o_code, binary, libs)

    if verbose:
        print(cmd)

    output = subprocess.check_output(cmd, shell=True)

    if verbose:
        print(output)

    # TODO shall we uncomment this?
#    # write and save a log file in .pyccel/'filename'.log
#    # ...
#    def mkdir_p(dir):
#        # type: (unicode) -> None
#        if os.path.isdir(dir):
#            return
#        os.makedirs(dir)
#
#    if True:
#        tmp_dir = '.pyccel'
#        mkdir_p(tmp_dir)
#        logfile = '{0}.log'.format(binary)
#        logfile = os.path.join(tmp_dir, logfile)
#        f = open(logfile, 'w')
#        f.write(output)
#        f.close()

    return output, cmd
    # ...
# ...

def execute_pyccel(filename,
                   compiler='gfortran',
                   debug=False,
                   accelerator=None,
                   include=[],
                   libdir=[],
                   modules=[],
                   libs=[],
                   binary=None):
    """Executes the full process:
        - parsing the python code
        - annotating the python code
        - converting from python to fortran
        - compiling the fortran code.

    """
    pyccel = Parser(filename)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    code = codegen.doprint()
    fname = codegen.export()

    # ... constructs the compiler flags
    flags = construct_flags(compiler,
                            debug=debug,
                            accelerator=accelerator,
                            include=include,
                            libdir=libdir)
    # ...

    # ... compile fortran code
    output, cmd = compile_fortran(fname, compiler, flags,
                                  binary=binary,
                                  verbose=False,
                                  modules=modules,
                                  is_module=codegen.is_module,
                                  libs=libs)
    # ...


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    execute_pyccel(filename)
