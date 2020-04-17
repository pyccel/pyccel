# coding: utf-8

#$ header class Point(public)
#$ header method __init__(Point, [double])
#$ header method __del__(Point)
#$ header method translate(Point, [double])

class Point(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

    def translate(self, a):
        b      =  self.x + a
        self.x = b

x = [0.,0.,0.]
p = Point(x)

a = [1.,1.,1.]

p.translate(a)
print(p.x)

