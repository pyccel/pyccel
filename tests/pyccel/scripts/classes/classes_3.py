# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
#$ header class Point(public)
#$ header method __init__(Point, double, double)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

if __name__ == '__main__':
    p = Point(0.0, 0.0)
    x=p.x
    p.x=4
    a = p.x
    a = p.x - 2
    a = 2 * p.x - 2
    a = 2 * (p.x + 6) - 2

    p.y = a + 5
    p.y = p.x + 5

    print(p.x, p.y)
    print(x)
