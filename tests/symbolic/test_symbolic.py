# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import os
import pytest

import numpy as np
import sympy as sp

import mappings

from pyccel import lambdify as pyc_lambdify
from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

RTOL = 1e-14
ATOL = 1e-15

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [f for f in files if f.endswith(".py")]

@pytest.mark.parametrize( "f", files )
@pytest.mark.skip(reason="Broken symbolic function support, see issue #330")
def test_symbolic(f):

    pyccel = Parser(f)
    pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name, 'fortran')
    codegen.printer.doprint(codegen.ast)

    # reset Errors singleton
    errors = Errors()
    errors.reset()

def test_lambdify(language):
    r1 = np.linspace(0.0, 1.0, 100)
    p1 = np.linspace(0.0, 2*np.pi, 100)
    r,p = np.meshgrid(r1, p1)
    x,y = sp.symbols('x1,x2')
    for m in (mappings.PolarMapping, mappings.TargetMapping, mappings.CzarnyMapping):
        expr_x = sp.sympify(m.expressions['x']).subs(m.constants)
        expr_y = sp.sympify(m.expressions['y']).subs(m.constants)
        sp_x = sp.lambdify([x, y], expr_x)
        sp_y = sp.lambdify([x, y], expr_y)
        pyc_x = pyc_lambdify(expr_x, {x : 'float[:,:]', y : 'float[:,:]'}, result_type = 'float[:,:]',
                    language = language)
        pyc_y = pyc_lambdify(expr_y, {x : 'float[:,:]', y : 'float[:,:]'}, result_type = 'float[:,:]',
                    language = language)

        print("Abs err:", np.abs(sp_x(r, p) - pyc_x(r, p)).max(), "<", ATOL)
        assert np.allclose(sp_x(r, p), pyc_x(r, p), rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_y(r, p), pyc_y(r, p), rtol=RTOL, atol=ATOL)

        pyc_x = pyc_lambdify(expr_x, {x : 'T', y : 'T'}, templates = {'T': ['float[:]', 'float[:,:]']},
                    language = language)
        pyc_y = pyc_lambdify(expr_y, {x : 'T', y : 'T'}, templates = {'T': ['float[:]', 'float[:,:]']},
                    language = language)

        print("Abs err:", np.abs(sp_x(r, p) - pyc_x(r, p)).max(), "<", ATOL)
        print("Rel err:", np.min(np.where(np.abs(pyc_x(r, p))== 0, RTOL, RTOL*np.abs(pyc_x(r, p)))))
        assert np.allclose(sp_x(r, p), pyc_x(r, p), rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_y(r, p), pyc_y(r, p), rtol=RTOL, atol=ATOL)

        assert np.allclose(sp_x(r[0,:], p[0,:]), pyc_x(r[0,:], p[0,:]), rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_y(r[0,:], p[0,:]), pyc_y(r[0,:], p[0,:]), rtol=RTOL, atol=ATOL)

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***      TESTING SYMBOLIC     ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print(f'> testing {f}')
        test_symbolic(f)
        print('\n')
