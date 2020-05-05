# coding: utf-8
#$ header class Matrix(public)
#$ header method __init__(Matrix, int, int)
#$ header method __del__(Matrix)
#$ header method add(Matrix,Matrix)
#$ header method dot(Matrix,Matrix)
#$ header method get(Matrix)
#$ header method set(Matrix,int,int,double)



class Matrix(object):
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.a = zeros((n_rows, n_cols), double)

    def __del__(self):
        del self.a
    def add(self,mat):
        self.a=self.a+mat.a

    def dot(self,mat):
        if self.n_cols==mat.n_rows:
            C=zeros((self.n_rows, mat.n_cols), double)
            for i in range(0,self.n_rows):
                for j in range(0,mat.n_cols):
                    s=0
                    for k in range(0,self.n_cols):
                        s=s+self.a[i,k]*mat.a[k,j]
                    C[i,j]=s



p = Matrix(2,3)
k = Matrix(2,3)
d = p.n_rows
p.n_rows = 5
s=p.a[0,0]
k.add(p)
del p
