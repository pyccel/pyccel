import subprocess
import os
import importlib
import numpy as np
import pytest

@pytest.mark.xfail
def test_imports():
    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    os.chdir(path_dir)

    test_file = "test_imports.py"

    filename = test_file[:-3]
    val = subprocess.check_call("python3 '%s' > test.ref" % test_file,   shell=True) 
    val = subprocess.check_call("pyccel '%s'" % test_file,   shell=True) 
    print("./%s > test.out" % filename)
    val = subprocess.check_call("./%s > test.out" % filename,   shell=True) 
    assert(np.loadtxt("test.out") == np.loadtxt("test.ref"))

def test_funcs():
    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    os.chdir(path_dir)

    test_file = "test_funcs.py"

    filename = test_file[:-3]
    val = subprocess.check_call("python3 '%s' > test.ref" % test_file,   shell=True) 
    val = subprocess.check_call("pyccel '%s'" % test_file,   shell=True) 
    print("./%s > test.out" % filename)
    val = subprocess.check_call("./%s > test.out" % filename,   shell=True) 
    assert(np.loadtxt("test.out") == np.loadtxt("test.ref"))

