# coding: utf-8

# TODO fix is. not working yet

from pyccel.codegen import load_module
import os

dir_path     = os.path.dirname(os.path.realpath(__file__))
scripts_path = os.path.join(dir_path, "../../scripts")

def test_helloworld():
    # ...
    filename = os.path.join(scripts_path, "helloworld.py")
    module = load_module(filename=filename)
    module.helloworld()
    # ...

def test_incr():
    # ...
    module = load_module(filename="../../scripts/incr.py")
    module.incr(3)
    module.decr(4)
    # ...

###################################
if __name__ == '__main__':
    test_helloworld()
#    test_incr()
