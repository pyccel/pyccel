# coding: utf-8

import os

from pyccel.codegen.utilities import load_extension
from pyccel.codegen.cmake import CMake

def test_cmake(src_dir, prefix=os.environ['CLAPP_DIR']):
    if not os.path.exists(src_dir):
        raise ValueError('Could not find :{0}'.format(src_dir))

    try:

#        FLAGS  = self.configs['flags']
#
#        FC     = self.configs['fortran']['compiler']
#        FFLAGS = self.configs['fortran']['flags']
#
#        cmake = CMake(project, src_dir, \
#                      prefix=self.prefix, \
#                      flags=FLAGS, \
#                      flags_fortran=FFLAGS, \
#                      compiler_fortran=FC)

        cmake = CMake(src_dir, prefix=prefix)

#        cmake.configure()
#        cmake.make()
#        cmake.install()

    except Exception as e:
        print(e)

#####################################
if __name__ == '__main__':
    base_dir = 'poisson'
    os.system('mkdir -p {0}'.format(base_dir))

    pwd = os.getcwd()
    os.chdir(base_dir)

#    test_cmake('extensions')
    load_extension('math', 'extensions', silent=False)

    os.chdir(pwd)

