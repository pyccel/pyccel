# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# TODO test if compiler exists before execute_pyccelning mpi, openacc
#      execute the binary file

import os
import pytest
from pyccel.codegen.pipeline import execute_pyccel

def get_files_from_folder(foldername):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join('scripts',foldername))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files

@pytest.mark.parametrize("f", get_files_from_folder('blas'))
@pytest.mark.external
def test_blas(f):
    execute_pyccel(f, libs=['blas'])

@pytest.mark.parametrize("f", get_files_from_folder('lapack'))
@pytest.mark.external
def test_lapack(f):
    execute_pyccel(f, libs=['blas', 'lapack'])

@pytest.mark.parametrize("f", get_files_from_folder('mpi'))
@pytest.mark.external
def test_mpi(f):
    execute_pyccel(f, accelerators=['mpi'])

@pytest.mark.parametrize("f", get_files_from_folder('openmp'))
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
@pytest.mark.external
def test_openmp(f, language):
    execute_pyccel(f, accelerators=['openmp'], language=language)

@pytest.mark.parametrize("f", get_files_from_folder('ccuda'))
@pytest.mark.parametrize( 'language',
        (pytest.param("ccuda", marks = pytest.mark.ccuda),)
)
@pytest.mark.external
def test_ccuda(f, language):
    execute_pyccel(f, language=language, verbose=True)

#@pytest.mark.parametrize("f", get_files_from_folder('openacc'))
#@pytest.mark.external
#def test_openacc():
#    execute_pyccel(f, compiler='pgfortran', accelerator='openacc')


######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/BLAS    ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('blas'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_blas(f)
    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/LAPACK  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('lapack'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_lapack(f)
    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/MPI     ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('mpi'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_mpi(f)
    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/OPENMP  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('openmp'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_openmp(f)
    print('\n')

#    print('*********************************')
#    print('***                           ***')
#    print('***  TESTING INTERNAL/OPENACC ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('openacc'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_openacc(f)
#    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING INTERNAL/Cuda ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('ccuda'):
       print('> testing {0}'.format(str(os.path.basename(f))))
       test_ccuda(f)
    print('\n')
