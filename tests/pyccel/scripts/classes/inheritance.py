# Static class methods are not yet supported
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class Point2d:
    def __init__(self : 'Point2d', x : float, y : float):
        self.x = x
        self.y = y

    def translate(self : 'Point2d', a : float, b : float):
        self.x = self.x + a
        self.y = self.y + b

    def as_tuple(self):
        return (self.x, self.y)

class Point3d(Point2d):
    def __init__(self : 'Point3d', x : float, y : float, z : float):
        self.z = z
        Point2d.__init__(self,x, y)

    def translate(self : 'Point3d', a : float, b : float, c : float):
        self.z = self.z + c
        super().translate(a,b)

    def as_tuple(self):
        return (self.x, self.y, self.z)

if __name__ == '__main__':
    p = Point2d(0.0, 0.0)
    p.translate(1.0, 2.0)
    p1= Point3d(0.0, 0.0, 0.0)
    p1.translate(1.,2.,3.)

    print(p.as_tuple())
    print(p1.as_tuple())
