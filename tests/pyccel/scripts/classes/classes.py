# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
#$ header class Point(public)
#$ header method __init__(Point, double, double)
#$ header method __del__(Point)
#$ header method translate(Point, double, double)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __del__(self):
        pass

    def translate(self, a, b):
        self.x = self.x + a
        self.y = self.y + b

if __name__ == '__main__':
    p = Point(0.0, 0.0)
    x=p.x
    p.x=x
    a = p.x
    a = p.x - 2
    a = 2 * p.x - 2
    a = 2 * (p.x + 6) - 2

    p.y = a + 5
    p.y = p.x + 5

    p.translate(1.0, 2.0)

    print(p.x, p.y)
    print(a)

    del p
