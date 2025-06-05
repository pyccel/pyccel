# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8

class Line(object):
    def __init__(self : 'Line', a : int, b : int, step : int):
        self.a = a
        self.b = b
        self.step = step

        self.indices = range(self.a, self.b, self.step)

    def __del__(self : 'Line'):
        pass

p = Line (0, 5, 1)

for i in p.indices:
    print(i)

del p
