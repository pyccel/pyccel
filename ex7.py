# coding: utf-8

#$ header class Matrix(public)
#$ header method __init__(Matrix, int, int)
#$ header method __del__(Matrix)

class Matrix(object):
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.a = zeros((n_rows, n_cols), double)

    def __del__(self):
        x = 0

