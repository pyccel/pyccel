# coding: utf-8

#$ header class Matrix(public)
#$ header method __init__(Matrix, double [:,:])
#$ header method __del__(Matrix)
#$ header method dot(Matrix, double [:])

class Matrix(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

    def dot(self, v):
        a = zeros_like(v)

x = ones((3,3), double)
p = Matrix (x)

del p
