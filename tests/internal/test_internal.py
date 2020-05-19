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
    print('> testing {0}'.format(str(os.path.basename(f))))

    execute_pyccel(f, libs=['blas'])

    print('\n')

@pytest.mark.parametrize("f", get_files_from_folder('lapack'))
def test_lapack(f):
    print('> testing {0}'.format(str(os.path.basename(f))))

    execute_pyccel(f, libs=['blas', 'lapack'])

    print('\n')

@pytest.mark.parametrize("f", get_files_from_folder('mpi'))
def test_mpi(f):
    print('> testing {0}'.format(str(os.path.basename(f))))

    execute_pyccel(f, compiler='mpif90')

    print('\n')

@pytest.mark.parametrize("f", get_files_from_folder('openmp'))
def test_openmp(f):
    print('> testing {0}'.format(str(os.path.basename(f))))

    execute_pyccel(f, accelerator='openmp')

    print('\n')

#@pytest.mark.parametrize("f", get_files_from_folder('openacc'))
#def test_openacc():
#    print('> testing {0}'.format(str(os.path.basename(f))))
#
#    execute_pyccel(f, compiler='pgfortran', accelerator='openacc')
#
#    print('\n')


######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/BLAS    ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('blas'):
        test_blas(f)

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/LAPACK  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('lapack'):
        test_lapack(f)

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/MPI     ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('mpi'):
        test_mpi(f)

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/OPENMP  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('openmp'):
        test_openmp(f)

#    print('*********************************')
#    print('***                           ***')
#    print('***  TESTING INTERNAL/OPENACC ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('openacc'):
#        test_openacc(f)
