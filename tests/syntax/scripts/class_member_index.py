# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# coding: utf-8

class Square(object):
    def __init__(self : 'Square', starts : 'int[:]', stops : 'int[:]', steps : 'int[:]'):
        self.starts = starts
        self.stops  = stops
        self.steps  = steps

        self.rx = range(self.starts[0], self.stops[0], self.steps[0])
        self.ry = range(self.starts[1], self.stops[1], self.steps[1])

        self.indices = tensor (self.rx, self.ry)

    def __del__(self : 'Square'):
        pass

starts = zeros(2, int)
stops  = zeros(2, int)
steps  = ones(2, int)

stops[0] = 4
stops[1] = 5

p = Square (starts, stops, steps)

for i,j in p.indices:
    print(i,j)

del p
