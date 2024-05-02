# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from sympy import Symbol
from sympy.sets.sets import FiniteSet
from pyccel.ast.parallel.communicator import Communicator, split
from pyccel.ast.parallel.group        import Group, Range, UniversalGroup, Split
from pyccel.ast.parallel.group        import Union, Intersection, Difference

g1 = Group(1, 3, 6, 7, 8, 9)
g2 = Group(0, 2, 4, 5, 8, 9)
print("g1      = ", g1)
print("g2      = ", g2)
print("g1.size = ", g1.size)
print("g2.size = ", g2.size)

union = Union(g1, g2)
print("union      = ", union)
print("union.size = ", union.size)

intersection = Intersection(g1, g2)
print("intersection      = ", intersection)
print("intersection.size = ", intersection.size)

difference = Difference(g1, g2)
print("difference      = ", difference)
print("difference.size = ", difference.size)

n = Symbol('n')
g3 = Range(0, 4)
print("g3 = ", g3)

gw = UniversalGroup()
print("gw.processes = ", gw.processes)
commw = gw.communicator

print("g1.communicator = ", g1.communicator)
comm1 = Communicator(g1)
print("comm1 = ", comm1)
print("g1.communicator = ", g1.communicator)

colors = [1, 1, 0, 1, 0, 0]
g4 = Split(gw, colors, 0)
comm4 = split(commw, g4)
print("comm4 = ", comm4)
g4 = comm4.group
print("g4 = ", g4)


