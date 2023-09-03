# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring, no-self-argument
# coding: utf-8

#$ header class Point(public)
#$ header method __init__(Point, double[:])
#$ header method __del__(Point)
#$ header method translate(Point, double[:])
import numpy as np
class Point(object):
    def __init__(this, x):
        this.x = x

    def __del__(self):
        pass

    def translate(this, a):
        this.x[:]   =  this.x + a

if __name__ == '__main__':
    x = np.array([0.,0.,0.])
    p = Point(x)

    a = np.array([1.,1.,1.])

    p.translate(a)
    print(p.x)

