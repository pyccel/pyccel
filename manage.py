# coding: utf-8

import os

from pyccel.codegen.utilities import load_extension
from pyccel.codegen.cmake import CMake



def test_cmake(base_dir, project, suffix, libname, prefix=os.environ['CLAPP_DIR']):
    if not os.path.exists(base_dir):
        raise ValueError('Could not find :{0}'.format(base_dir))

    FC     = 'gfortran'
    FLAGS  = {}
    FFLAGS = '-O2 -fbounds-check'

    cmake = CMake(base_dir, \
                  prefix=prefix, \
                  flags=FLAGS, \
                  flags_fortran=FFLAGS, \
                  compiler_fortran=FC)

    cmake.initialize(base_dir, project, suffix, libname, force=True)

    cmake.configure()
    cmake.make()
    cmake.install()


#        FLAGS  = self.configs['flags']
#
#        FC     = self.configs['fortran']['compiler']
#        FFLAGS = self.configs['fortran']['flags']


#####################################
if __name__ == '__main__':
    project = 'poisson'
    suffix  = 'pss'
    libname = 'poisson'

    base_dir = 'poisson'
    os.system('mkdir -p {0}'.format(base_dir))

    pwd = os.getcwd()
    os.chdir(base_dir)

    load_extension('math', 'extensions', silent=False)

    os.chdir(pwd)

    test_cmake(base_dir, project, suffix, libname)
