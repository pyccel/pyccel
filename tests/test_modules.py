# coding: utf-8

from pyccel.codegen import load_module

def test_helloworld():
    # ...
    module = load_module(filename="helloworld.py")
    module.helloworld()
    # ...

def test_incr():
    # ...
    module = load_module(filename="incr.py")
    module.incr(3)
    module.decr(4)
    # ...

###################################
if __name__ == '__main__':
#    test_helloworld()
    test_incr()
