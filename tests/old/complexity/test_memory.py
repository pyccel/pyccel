# coding: utf-8

from pyccel.complexity.memory import MemComplexity
from os.path import join, dirname

this_folder = dirname(__file__)
inputs = join(this_folder, "inputs")

def test_1():
    filename = 'ex1.py'
    print(("==== " + filename + " ===="))
    filename = join(inputs, filename)
    complexity = MemComplexity(filename)
    print(">>> computational intensity ~", \
            complexity.intensity())

def test_2():
    filename = 'ex2.py'
    print(("==== " + filename + " ===="))
    filename = join(inputs, filename)
    complexity = MemComplexity(filename)
    print(">>> computational intensity ~", \
            complexity.intensity(local_vars=['r']))

def test_3():
    filename = 'ex3.py'
    print(("==== " + filename + " ===="))
    filename = join(inputs, filename)
    complexity = MemComplexity(filename)
    print(">>> computational intensity ~", \
            complexity.intensity())

def test_4():
    filename = 'ex4.py'
    print(("==== " + filename + " ===="))
    filename = join(inputs, filename)
    complexity = MemComplexity(filename)
    print(">>> computational intensity ~", \
            complexity.intensity(local_vars=['u','v','r']))

##############################################
if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
