# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.complexity.memory import MemComplexity
import os

from sympy.abc import n,m,x,b
from sympy import Function, Symbol
from sympy import simplify as sp_simplify
SHAPE = Function('shape')
READ = Symbol('READ')
WRITE = Symbol('WRITE')
out = Symbol('out')
k1 = Symbol('k1')
k2 = Symbol('k2')
p1 = Symbol('p1')
p2 = Symbol('p2')
ne1 = Symbol('ne1')
ne2 = Symbol('ne2')

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

# ==============================================================================
def test_complexity_ex2():

    f = path_dir + '/ex2.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [READ + WRITE + n**2*(3*READ*(n - 1) + 8*READ + WRITE*(n - 1) + 6*WRITE)/2,# f1
            READ + WRITE + n ** 2 * (4 * READ * (n - 1) + 6 * READ + WRITE * (n - 1) + 4 * WRITE) / 2,# f2
            READ * n ** 3 / 2 + 3 * READ * n ** 2 / 2 + READ + WRITE * n ** 3 / 6 + 3 * WRITE * n ** 2 / 2 + WRITE * n / 3 + WRITE# f3
            ]


    for f, c in complexity.costs.items():
        assert sp_simplify(c - comp[i]) == 0
        i = i + 1



# ==============================================================================
def test_complexity_ex_assembly():

    f = path_dir + '/ex_assembly.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [2*READ + WRITE*k1*k2 + WRITE*(p1 + 1)*(p2 + 1) + 2*WRITE + ne1*(READ + WRITE + ne2*(READ*(p1 + 1)*(p2 + 1) + READ + WRITE*k1*k2 + WRITE*(p1 + 1)*(p2 + 1) + WRITE + (p1 + 1)**2*(p2 + 1)**2*(14*READ + 6*WRITE + k1*k2*(22*READ + 9*WRITE)) + (p1 + 1)*(p2 + 1)*(READ + WRITE + k1*(READ + WRITE + k2*(5*READ + 2*WRITE)))))
            ]


    for f, c in complexity.costs.items():
            assert sp_simplify(c - comp[i]) == 0
            i = i + 1




# ==============================================================================
def test_complexity_mxm():

    f = path_dir + '/mxm.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [READ + WRITE + n**3*(3*READ + WRITE),
            (b*(READ + 3*WRITE*b**2 + WRITE) + n**2*(2*b*(READ + WRITE) + n*(2*READ + 2*WRITE + b*(3*READ + WRITE))))/b
            ]


    for f, c in complexity.costs.items():
            assert sp_simplify(c - comp[i]) == 0
            i = i + 1



# ==============================================================================
def test_complexity_qr():

    f = path_dir + '/qr.py'
    mode = None

    complexity = MemComplexity(f)
    complexity.cost(mode=mode, simplify=True, bigo=None)
#    complexity.cost(mode=mode, simplify=True, bigo=['n'])

    print('----------------------')
    i = 0
    comp = [READ*n**3/2 + 3*READ*n**2/2 + READ + WRITE*n**3/6 + 3*WRITE*n**2/2 + WRITE*n/3 + WRITE,
            ]


    for f, c in complexity.costs.items():
            assert sp_simplify(c - comp[i]) == 0
            i = i + 1