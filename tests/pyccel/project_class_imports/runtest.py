# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from project.basics.Point_mod import Point
from project.basics.Line_mod import Line
from project.shapes.Square_mod import Square

if __name__ == '__main__':
    p1 = Point(0.0, 0.0)
    p2 = Point(1.0, 0.0)
    p3 = Point(0.0, 1.0)
    p4 = Point(1.0, 1.0)

    s = Square(p1, p2, p3, p4)

    print(s.corner_1)
    print(s.corner_2)
    print(s.corner_3)
    print(s.corner_4)

    l = Line(p1, p2)

    s_x, s_y = l.get_start()
    e_x, e_y = l.get_end()
    print(s_x, s_y)
    print(e_x, e_y)
    print(int(l.longer_than(0.5)))
    print(int(l.longer_than(2)))
