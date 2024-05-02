# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# TODO test if compiler exists before execute_pyccelning mpi, openacc
#      execute the binary file

from pyccel.codegen.pipeline import execute_pyccel
import os
import pytest

def get_files_from_folder(foldername):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join('scripts',foldername))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files

@pytest.mark.parametrize("f", get_files_from_folder('blas'))
def test_blas(f):
    execute_pyccel(f, libs=['blas'])

@pytest.mark.parametrize("f", get_files_from_folder('lapack'))
def test_lapack(f):
    execute_pyccel(f, libs=['blas', 'lapack'])

#@pytest.mark.parametrize("f", get_files_from_folder('MPI'))
#def test_mpi(f):
#    execute_pyccel(f, compiler='mpif90')
#
#@pytest.mark.parametrize("f", get_files_from_folder('openmp'))
#def test_openmp(f):
#    execute_pyccel(f, accelerator='openmp')
#
#@pytest.mark.parametrize("f", get_files_from_folder('openacc'))
#def test_openacc():
#    execute_pyccel(f, compiler='pgfortran', accelerator='openacc')


######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***  TESTING MACRO/BLAS    ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('blas'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_blas(f)
    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING MACRO/LAPACK  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('lapack'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_lapack(f)
    print('\n')

#    print('*********************************')
#    print('***                           ***')
#    print('***  TESTING MACRO/MPI     ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('MPI'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_mpi(f)
#    print('\n')
#
#    print('*********************************')
#    print('***                           ***')
#    print('***  TESTING MACRO/OPENMP  ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('openmp'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_openmp(f)
#    print('\n')
#
#    print('*********************************')
#    print('***                           ***')
#    print('***  TESTING MACRO/OPENACC ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('openacc'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_openacc(f)
#    print('\n')
