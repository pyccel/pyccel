# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
#$ header class Point(public)
#$ header method __init__(Point, double, double)
#$ header method __del__(Point)
#$ header method translate(Point, double|int, double|int)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __del__(self):
        pass

    def translate(self, a, b):
        self.x = self.x + a
        self.y = self.y + b
