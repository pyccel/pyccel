# coding: utf-8

#$ header class Matrix(abstract, public)
#$ header method __init__(Matrix, int, int)

class Matrix(object):
    def __init__(self, n_rows, n_cols):
        x = n_rows + n_cols
