# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

#$ header class Point1(public)
#$ header method __init__(Point1, double)
#$ header class Point2(public)
#$ header method __init__(Point2, double)
#$ header method test_func(Point2)

class Point1:
    def __init__(self, x):
        self.x = x

class Point2:
    def __init__(self, y):
        self.y = y

    def test_func(self):
        p = Point1(self.y)
        print(p.x)

if __name__ == '__main__':
    j = Point2(2.2)
    j.test_func()
