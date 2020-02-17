# coding: utf-8

# TODO test if compiler exists before running mpi, openacc
#      execute the binary file

from pyccel.codegen.pipeline import execute_pyccel
import os

def run(test_dir , **settings):
    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join('scripts', test_dir))

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py") and 'mpi4py' not in f)]
    
    os.chdir(path_dir)
    for f in files:
        print('> testing {0}'.format(str(f)))

        execute_pyccel(f, **settings)

    os.chdir(init_dir)
    print('\n')

def test_mpi4py():
    print('*********************************')
    print('***                           ***')
    print('***  TESTING EXTERNAL/MPI4PY  ***')
    print('***                           ***')
    print('*********************************')

    run('mpi4py', compiler='mpif90')


######################
if __name__ == '__main__':
    test_mpi4py()
