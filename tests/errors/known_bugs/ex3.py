# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

#$ header class Line(public)
#$ header method __init__(Line, int, int, int)
#$ header method __del__(Line)

class Line(object):
    def __init__(self, a, b, step):
        self.a = a
        self.b = b
        self.step = step

        self.indices = range(self.a, self.b, self.step)

    def __del__(self):
        pass

p = Line (0, 5, 1)

for i in p.indices:
    print(i)

del p
