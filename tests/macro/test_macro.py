# coding: utf-8

# TODO test if compiler exists before running mpi, openacc
#      execute the binary file

from pyccel.codegen.pipeline import execute_pyccel
import os

def run(test_dir, **settings):
    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join('scripts', test_dir))

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
        print('> testing {0}'.format(str(f)))

        execute_pyccel(f, **settings)

    os.chdir(init_dir)
    print('\n')

def test_blas():
    print('*********************************')
    print('***                           ***')
    print('***    TESTING MACRO/BLAS     ***')
    print('***                           ***')
    print('*********************************')

    run('blas', libs=['blas'])

def test_lapack():
    print('*********************************')
    print('***                           ***')
    print('***    TESTING MACRO/LAPACK   ***')
    print('***                           ***')
    print('*********************************')

    run('lapack', libs=['blas', 'lapack'])

#def test_mpi():
#    print('*********************************')
#    print('***                           ***')
#    print('***    TESTING MACRO/MPI      ***')
#    print('***                           ***')
#    print('*********************************')
#
#    run('mpi', compiler='mpif90')

#def test_openmp():
#    print('*********************************')
#    print('***                           ***')
#    print('***    TESTING MACRO/OPENMP   ***')
#    print('***                           ***')
#    print('*********************************')
#
#    run('openmp', accelerator='openmp')

#def test_openacc():
#    print('*********************************')
#    print('***                           ***')
#    print('***    TESTING MACRO/OPENACC  ***')
#    print('***                           ***')
#    print('*********************************')
#
#    run('openacc', compiler='pgfortran', accelerator='openacc')


######################
if __name__ == '__main__':
    test_blas()
    test_lapack()
#    test_mpi()
#    test_openmp()
#    test_openacc()
