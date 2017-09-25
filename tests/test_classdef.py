# coding: utf-8
from pyccel.types.ast import Assign, Variable, FunctionDef, ClassDef
x = Variable('float', 'x')
y = Variable('float', 'y')
n = Variable('int', 'n')
args        = [x, n]
results     = [y]
body        = [Assign(y,x+n)]
f = FunctionDef('f', args, results, body, [], [])
n_rows = Variable('int', 'n_rows')
n_cols = Variable('int', 'n_cols')
attributs   = [n_rows, n_cols]
methods     = [f]
print ClassDef('Matrix', attributs, methods)
