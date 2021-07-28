import json
import os
import sys
import sysconfig
from itertools import chain
from numpy import get_include as get_numpy_include
from pyccel import __version__ as pyccel_version

gfort_info = {'exec' : 'gfortran',
              'mpi_exec' : 'mpif90',
              'language': 'fortran',
              'module_output_flag': '-J',
              'debug_flags': ("-fcheck=bounds",),
              'release_flags': ("-O3",),
              'standard_flags' : ('-std=f2003','-fPIC'),
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
    gfort_info['mpi']['flags'] = ('-D','USE_MPI_MODULE')
    gfort_info['mpi']['includes'] = (os.environ["MSMPI_INC"].rstrip('\\'),)
    gfort_info['mpi']['libs'] = (os.environ["MSMPI_LIB64"].rstrip('\\'),)
    gfort_info['mpi']['dependencies'] = (os.path.join(os.environ["MSMPI_LIB64"], 'libmsmpi.a'),)

#------------------------------------------------------------
ifort_info = {'exec' : 'ifort',
              'mpi_exec' : 'mpiifort',
              'language': 'fortran',
              'module_output_flag': '-module',
              'debug_flags': ("-check=bounds",),
              'release_flags': ("-O3",),
              'standard_flags' : ('-std=f2003','-fPIC'),
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
gcc_info = {'exec' : 'gcc',
            'mpi_exec' : 'mpicc',
            'language': 'c',
            'debug_flags': ("-g",),
            'release_flags': ("-O3",),
            'standard_flags' : ('-std=c99','-fPIC'),
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
    gcc_info['openmp']['flags'] = ("-Xpreprocessor",'fopenmp')
    gcc_info['openmp']['libs'] = ('omp',)

#------------------------------------------------------------
icc_info = {'exec' : 'icc',
            'mpi_exec' : 'mpiicc',
            'language': 'c',
            'debug_flags': ("-g",),
            'release_flags': ("-O3",),
            'standard_flags' : ('-std=c99','-fPIC'),
            'openmp': {
                'flags' : ('-fopenmp',),
                },
            'openacc': {
                'flags' : ("-ta=multicore", "-Minfo=accel"),
                },
            'family': 'intel',
            }
#------------------------------------------------------------
config_vars = sysconfig.get_config_vars()
python_libs = config_vars.get("LIBPYTHON","").split() \
                    +config_vars.get("BLDLIBRARY","").split() \
                    +config_vars.get("LIBS","").split() \
                    +config_vars.get("LIBPL","").split()
python_info = {
        "libs" : [s[2:] for s in config_vars.get("LIBM","").split()], # Strip -l from beginning
        'python': {
            'flags' : config_vars.get("CFLAGS","").split()\
                + config_vars.get("CC","").split()[1:],
            'includes' : [*config_vars.get("INCLUDEPY","").split(), get_numpy_include()],
            'libs' : [l for l in python_libs if l.startswith('-l')],
            'libdirs' : [l[2:] for l in python_libs if l.startswith('-L')]+config_vars.get("LIBPL","").split(),
            "linker_flags" : config_vars.get("LDSHARED","").split()[1:],
            "shared_suffix" : config_vars.get("EXT_SUFFIX",".so"),
            }
        }
if sys.platform == "win32":
    python_info['python']['dependencies'] = ('python{}.lib'.format(config_vars["VERSION"]),)
    #python_info['python'] = config_vars.get("LIBDEST","").split()

save_folder = os.path.dirname(os.path.abspath(__file__))

def print_json(filename, info):
    print(json.dumps({k:v for k,v in chain(info.items(),
                                            python_info.items(),
                                            [('pyccel_version', pyccel_version)])},
                     indent=4),
          file=open(os.path.join(save_folder, filename),'w'))

def generate_default():
    files = {
            'gfortran.json' : gfort_info,
            'gcc.json'      : gcc_info,
            'ifort.json'    : ifort_info,
            'icc.json'      : icc_info
            }
    for f, d in files.items():
        print_json(f,d)
    return files.keys()
