# coding: utf-8

#$ header class Point(public)
#$ header method __init__(Point, double [:])
#$ header method __del__(Point)
#$ header method translate(Point, double [:])

class Point(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

    def translate(self, a):
        self.x = self.x + a

x = ones(3, double)
p = Point (x)

a = zeros(3, double)
a[0] = 3
p.translate(a)
print(p.x)

b = p.x[0]
b = p.x[0] + 1.0
b = 2 * p.x[0] + 1.0
b = 2 * ( p.x[0] + 1.0 )
print(b)

p.x[1] = 2.0

del p
