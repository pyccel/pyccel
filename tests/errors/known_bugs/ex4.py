# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

#$ header class Square(public)
#$ header method __init__(Square, int [:], int [:], int [:])
#$ header method __del__(Square)

class Square(object):
    def __init__(self, starts, stops, steps):
        self.starts = starts
        self.stops  = stops
        self.steps  = steps

        self.rx = range(self.starts[0], self.stops[0], self.steps[0])
        self.ry = range(self.starts[1], self.stops[1], self.steps[1])

        self.indices = tensor (self.rx, self.ry)

    def __del__(self):
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
