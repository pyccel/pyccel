# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

class Point:
    def __init__(self : 'Point', x : 'float', y : 'float'):
        self._coords = (x, y)

        self._x_tag = (x, True)
        self._y_tag = (x, False)

    def show_coords(self : 'Point'):
        print(self._coords)
        print(self._x_tag)

if __name__ == '__main__':
    p = Point(2.0, 3.0)
    p.show_coords()
    print(isinstance(p, Point))
