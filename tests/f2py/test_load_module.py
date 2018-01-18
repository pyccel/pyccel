# coding: utf-8

# TODO fix is. not working yet

from pyccel.codegen import load_module
import os

dir_path     = os.path.dirname(os.path.realpath(__file__))
scripts_path = os.path.join(dir_path, "scripts")

def test_helloworld():
    # ...
    filename = os.path.join(scripts_path, "helloworld.py")
    module = load_module(filename=filename)
    module.print_helloworld()
    # ...

def test_incr():
    # ...
    filename = os.path.join(scripts_path, "incrdecr.py")
    module = load_module(filename=filename)
    y = module.incr(3)
    y = module.decr(4)
    print ('> y = {}'.format(y))
    # ...

###################################
if __name__ == '__main__':
    test_helloworld()
    test_incr()
