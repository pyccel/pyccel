# coding: utf-8

# Usage:
#   python tests/test_pyccel_mpi.py --compiler=mpif90

# here the execution is handled by os.system, since some tests need specific
# number ot processors

import os

from pyccel.commands.console import pyccel
from utils import clean_tests

# ...
def test_core(n_procs=2):
    print('============== testing core ================')
    ignored = ['poisson_fd_1d.py', 'scatter.py']

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts/mpi/core')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    # we give here tests that only works with a given number of procs,
    d_tests = {'sendrecv.py': 2,
               'sendrecv_replace.py': 2,
               'line.py': 2,
               'column.py': 2,
               'cart2d_1.py': 4,
               'cart2d_2.py': 4}

    for f in files:
        f_name = os.path.join(path_dir, f)

        # we only convert and compile the generated code
        pyccel(files=[f_name], openmp=False)

        # then we use 'mpirun'
        binary = f_name.split('.py')[0]
        _n_procs = n_procs
        if f in d_tests:
            _n_procs = d_tests[f]
        cmd = 'mpirun -n {n_procs} {binary}'.format(n_procs=_n_procs,
                                                    binary=binary)

        print('> {0}'.format(cmd))
        os.system(cmd)

        print('> testing {0}: done'.format(str(f)))
# ...

# ...
def test_examples(n_procs=2):
    print('============== testing examples ================')
    ignored = []

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts/mpi/')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    # we give here tests that only works with a given number of procs,
    d_tests = {}

    for f in files:
        f_name = os.path.join(path_dir, f)

        # we only convert and compile the generated code
        pyccel(files=[f_name], openmp=False)

        # then we use 'mpirun'
        binary = f_name.split('.py')[0]
        _n_procs = n_procs
        if f in d_tests:
            _n_procs = d_tests[f]
        cmd = 'mpirun -n {n_procs} {binary}'.format(n_procs=_n_procs,
                                                    binary=binary)

        print('> {0}'.format(cmd))
        os.system(cmd)

        print('> testing {0}: done'.format(str(f)))
# ...

################################
if __name__ == '__main__':
    clean_tests()
    test_core()
    test_examples()
    clean_tests()
