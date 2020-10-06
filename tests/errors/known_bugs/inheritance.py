# pylint: disable=missing-function-docstring, missing-module-docstring/
#$ header class Point2d(public)
#$ header method __init__(Point2d, double, double)
#$ header method translate(Point2d, double, double)

class Point2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def translate(self, a, b):
        self.x = self.x + a
        self.y = self.y + b

#$ header class Point3d(public)
#$ header method __init__(Point3d, double, double, double)
#$ header method translate(Point3d, double, double, double)

class Point3d(Point2d):
    def __init__(self, x, y, z):
        self.z = z
        Point2d.__init__(self,x, y)
        #super().__init__(x, y)

    def translate(self, a, b, c):
        self.z = self.z + c
        Point2d.translate(self,a,b)
        #super().translate(a,b)

p = Point2d(0.0, 0.0)
p.translate(1.0, 2.0)
p1= Point3d(0.0, 0.0, 0.0)
p1.translate(1.,2.,3.)


