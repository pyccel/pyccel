import json
import os
import sys
import sysconfig
from itertools import chain
from numpy import get_include as get_numpy_include

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
            }
#------------------------------------------------------------
python_info = {
        "libs" : [s[2:] for s in sysconfig.get_config_var("LIBM").split()], # Strip -l from beginning
        'python': {
            'flags' : sysconfig.get_config_var("CFLAGS").split()\
                + sysconfig.get_config_var("CC").split()[1:],
            'includes' : [*sysconfig.get_config_var("INCLUDEPY").split(), get_numpy_include()],
            'libs' : [s[2:] for s in sysconfig.get_config_var("LIBPYTHON").split()
                                        +sysconfig.get_config_var("BLDLIBRARY").split()
                                        +sysconfig.get_config_var("LIBS").split()], #Strip -l from beginning
            'libdirs' : sysconfig.get_config_var("LIBPL").split(),
            "linker_flags" : sysconfig.get_config_var("LDSHARED").split()[1:],
            "shared_suffix" : sysconfig.get_config_var("EXT_SUFFIX"),
            }
        }

save_folder = os.path.dirname(os.path.abspath(__file__))

def print_json(filename, info):
    print(json.dumps({k:v for k,v in chain(info.items(), python_info.items())},
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
