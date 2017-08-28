# coding: utf-8

from pyccel.codegen import load_module

def test_helloworld():
    # ...
    module = load_module(filename="helloworld.py")
    module.helloworld()
    # ...

###################################
if __name__ == '__main__':
    test_helloworld()
