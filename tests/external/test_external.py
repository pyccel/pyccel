# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

# TODO test if compiler exists before running mpi, openacc
#      execute the binary file

import pytest

from pyccel.codegen.pipeline import execute_pyccel
import os

def get_files_from_folder(foldername):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join('scripts',foldername))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files

@pytest.mark.xfail(reason="Broken mpi4py support, see issue #251")
@pytest.mark.parametrize("f", get_files_from_folder('mpi4py'))
@pytest.mark.external
def test_mpi4py(f):

    execute_pyccel(f, compiler='mpif90')

    print('\n')

@pytest.mark.parametrize("f", get_files_from_folder('lapack'))
@pytest.mark.external
def test_lapack(f):

    execute_pyccel(f)

    print('\n')


######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***  TESTING EXTERNAL/MPI4PY  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('mpi4py'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_mpi4py(f)
