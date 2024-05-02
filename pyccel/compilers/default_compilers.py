"""
Module responsible for the creation of the json files containing the default configuration for each available compiler.
This module only needs to be imported once. Once the json files have been generated they can be used directly thus
avoiding the need for a large number of imports
"""
import os
import sys
import sysconfig
from numpy import get_include as get_numpy_include
from pyccel import __version__ as pyccel_version

gfort_info = {'exec' : 'gfortran',
              'mpi_exec' : 'mpif90',
              'language': 'fortran',
              'module_output_flag': '-J',
              'debug_flags': ("-fcheck=bounds",),
              'release_flags': ("-O3","-funroll-loops",),
              'general_flags' : ('-fPIC',),
              'standard_flags' : ('-std=f2003',),
              'mpi': {
                  },
              'openmp': {
                  'flags' : ('-fopenmp',),
                  'libs'  : ('gomp',),
                  },
              'openacc': {
                  'flags' : ("-ta=multicore", "-Minfo=accel"),
                  },
              'family': 'GNU',
              }
if sys.platform == "win32":
    gfort_info['mpi_exec'] = 'gfortran'
    gfort_info['mpi']['flags']    = ('-D','USE_MPI_MODULE')
    gfort_info['mpi']['libs']     = ('msmpi',)
    gfort_info['mpi']['includes'] = (os.environ["MSMPI_INC"].rstrip('\\'),)
    gfort_info['mpi']['libdirs']  = (os.environ["MSMPI_LIB64"].rstrip('\\'),)

#------------------------------------------------------------
ifort_info = {'exec' : 'ifort',
              'mpi_exec' : 'mpiifort',
              'language': 'fortran',
              'module_output_flag': '-module',
              'debug_flags': ("-check=bounds",),
              'release_flags': ("-O3","-funroll-loops",),
              'general_flags' : ('-fPIC',),
              'standard_flags' : ('-std=f2003',),
              'openmp': {
                  'flags' : ('-fopenmp','-nostandard-realloc-lhs'),
                  'libs'  : ('iomp5',),
                  },
              'openacc': {
                  'flags' : ("-ta=multicore", "-Minfo=accel"),
                  },
              'family': 'intel',
              }

#------------------------------------------------------------
pgfortran_info = {'exec' : 'pgfortran',
              'mpi_exec' : 'pgfortran',
              'language': 'fortran',
              'module_output_flag': '-module',
              'debug_flags': ("-Mbounds",),
              'release_flags': ("-O3","-Munroll",),
              'general_flags' : ('-fPIC',),
              'standard_flags' : ('-Mstandard',),
              'openmp': {
                  'flags' : ('-mp',),
                  },
              'openacc': {
                  'flags' : ("-acc"),
                  },
              'family': 'PGI',
              }

#------------------------------------------------------------
nvfort_info = {'exec' : 'nvfort',
              'mpi_exec' : 'nvfort',
              'language': 'fortran',
              'module_output_flag': '-module',
              'debug_flags': ("-Mbounds",),
              'release_flags': ("-O3","-Munroll",),
              'general_flags' : ('-fPIC',),
              'standard_flags' : ('-Mstandard',),
              'openmp': {
                  'flags' : ('-mp',),
                  },
              'openacc': {
                  'flags' : ("-acc"),
                  },
              'family': 'nvidia',
              }

#------------------------------------------------------------
gcc_info = {'exec' : 'gcc',
            'mpi_exec' : 'mpicc',
            'language': 'c',
            'debug_flags': ("-g",),
            'release_flags': ("-O3","-funroll-loops",),
            'general_flags' : ('-fPIC',),
            'standard_flags' : ('-std=c99',),
            'mpi': {
                },
            'openmp': {
                'flags' : ('-fopenmp',),
                'libs'  : ('gomp',),
                },
            'openacc': {
                'flags' : ("-ta=multicore", "-Minfo=accel"),
                },
            'family': 'GNU',
            }
if sys.platform == "darwin":
    gcc_info['openmp']['flags'] = ("-Xpreprocessor",'-fopenmp')
    gcc_info['openmp']['libs'] = ('omp',)
elif sys.platform == "win32":
    gcc_info['mpi_exec'] = 'gcc'
    gcc_info['mpi']['flags']    = ('-D','USE_MPI_MODULE')
    gcc_info['mpi']['libs']     = ('msmpi',)
    gcc_info['mpi']['includes'] = (os.environ["MSMPI_INC"].rstrip('\\'),)
    gcc_info['mpi']['libdirs']  = (os.environ["MSMPI_LIB64"].rstrip('\\'),)

#------------------------------------------------------------
icc_info = {'exec' : 'icc',
            'mpi_exec' : 'mpiicc',
            'language': 'c',
            'debug_flags': ("-g",),
            'release_flags': ("-O3","-funroll-loops",),
            'general_flags' : ('-fPIC',),
            'standard_flags' : ('-std=c99',),
            'openmp': {
                'flags' : ('-fopenmp',),
                },
            'openacc': {
                'flags' : ("-ta=multicore", "-Minfo=accel"),
                },
            'family': 'intel',
            }

#------------------------------------------------------------
pgcc_info = {'exec' : 'pgcc',
            'mpi_exec' : 'pgcc',
            'language': 'c',
            'debug_flags': ("-g",),
            'release_flags': ("-O3","-Munroll",),
            'general_flags' : ('-fPIC',),
            'standard_flags' : ('-std=c99',),
            'openmp': {
                'flags' : ('-mp',),
                },
            'openacc': {
                'flags' : ("-acc"),
                },
            'family': 'PGI',
            }

#------------------------------------------------------------
nvc_info = {'exec' : 'nvc',
            'mpi_exec' : 'nvc',
            'language': 'c',
            'debug_flags': ("-g",),
            'release_flags': ("-O3","-Munroll",),
            'general_flags' : ('-fPIC',),
            'standard_flags' : ('-std=c99',),
            'openmp': {
                'flags' : ('-mp',),
                },
            'openacc': {
                'flags' : ("-acc"),
                },
            'family': 'nvidia',
            }

#------------------------------------------------------------
def change_to_lib_flag(lib):
    """
    Convert a library to a library flag
    """
    if lib.startswith('lib'):
        end = len(lib)
        if lib.endswith('.a'):
            end = end-2
        if lib.endswith('.so'):
            end = end-3
        return '-l{}'.format(lib[3:end])
    else:
        return lib

python_version = sysconfig.get_python_version()
config_vars = sysconfig.get_config_vars()
linker_flags = [change_to_lib_flag(l) for l in
                    config_vars.get("LIBRARY","").split() + \
                    config_vars.get("LDSHARED","").split()[1:]]
python_info = {
        "libs" : config_vars.get("LIBM","").split(), # Strip -l from beginning
        "libdirs" : config_vars.get("LIBDIR","").split(),
        'python': {
            'flags' : config_vars.get("CFLAGS","").split()\
                + config_vars.get("CC","").split()[1:],
            'includes' : [*config_vars.get("INCLUDEPY","").split(), get_numpy_include()],
            'libs' : [l[2:] for l in linker_flags if l.startswith('-l')],
            'libdirs' : [l[2:] for l in linker_flags if l.startswith('-L')]+config_vars.get("LIBPL","").split(),
            "shared_suffix" : config_vars.get("EXT_SUFFIX",".so"),
            }
        }
if sys.platform == "win32":
    python_info['python']['libs'].append('python{}'.format(config_vars["VERSION"]))
    python_info['python']['libdirs'].extend(config_vars.get("installed_base","").split())

#------------------------------------------------------------
gcc_info.update(python_info)
gfort_info.update(python_info)
icc_info.update(python_info)
ifort_info.update(python_info)
pgcc_info.update(python_info)
pgfortran_info.update(python_info)
nvc_info.update(python_info)
nvfort_info.update(python_info)

available_compilers = {('GNU', 'c') : gcc_info,
                       ('GNU', 'fortran') : gfort_info,
                       ('intel', 'c') : icc_info,
                       ('intel', 'fortran') : ifort_info,
                       ('PGI', 'c') : pgcc_info,
                       ('PGI', 'fortran') : pgfortran_info,
                       ('nvidia', 'c') : nvc_info,
                       ('nvidia', 'fortran') : nvfort_info}

vendors = ('GNU','intel','PGI','nvidia')
