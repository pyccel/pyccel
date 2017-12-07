# coding: utf-8

import os

from pyccel.codegen.utilities import load_extension
from pyccel.codegen.cmake import CMake

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

# ...
def initialize_cmake(src_dir, project, suffix, libname, force=True):
    if not os.path.exists(src_dir):
        raise ValueError('Could not find :{0}'.format(src_dir))

    # ...
    from pyccel import codegen
    codegen_dir  = os.path.dirname(os.path.realpath(str(codegen.__file__)))
    templates_dir = os.path.join(codegen_dir, 'templates')

    cmakemodules_src = os.path.join(templates_dir, 'CMakeModules')
    cmakelists_src   = os.path.join(templates_dir, 'CMakeLists.txt')
    package_src      = os.path.join(templates_dir, 'package')

    cmakemodules_dst = os.path.join(src_dir, 'CMakeModules')
    cmakelists_dst   = os.path.join(src_dir, 'CMakeLists.txt')
    package_dst      = os.path.join(src_dir, 'package')
    # ...

    # ... update CMakeLists.txt
    def _print_cmakelists(src, dst):
        f = open(src, 'r')
        code = f.readlines()
        f.close()

        code = ''.join(l for l in code)

        code = code.replace('__PROJECT__', project)
        code = code.replace('__SRC_DIR__', src_dir)
        code = code.replace('__SUFFIX__',  suffix)
        code = code.replace('__LIBNAME__', libname)

        if force or (not os.path.isfile(dst)):
            f = open(dst, 'w')
            f.write(code)
            f.close()
    # ...

    # ...
    _print_cmakelists(cmakelists_src, cmakelists_dst)
    # ...

    # ...
    copyanything(cmakemodules_src, cmakemodules_dst)
    copyanything(package_src, package_dst)
    # ...

    # ...
    src = os.path.join(package_src, 'CMakeLists.txt')
    dst = os.path.join(package_dst, 'CMakeLists.txt')
    _print_cmakelists(src, dst)
    # ...
# ...

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
    project = 'poisson'
    suffix  = 'pss'
    libname = 'poisson'

    base_dir = 'poisson'
    os.system('mkdir -p {0}'.format(base_dir))

    pwd = os.getcwd()
    os.chdir(base_dir)

#    test_cmake('extensions')
    load_extension('math', 'extensions', silent=False)

    os.chdir(pwd)

    initialize_cmake(base_dir, project, suffix, libname, force=True)
