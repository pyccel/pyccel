# coding: utf-8

#$ header class    Matrix(abstract, public)
#$ header function __init__(Matrix, int, int)
#$ header function dot(Matrix, double [:])

class Matrix(object):

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def dot(self, x):
        y = x
        return y
