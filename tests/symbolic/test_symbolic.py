# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import os
from typing import TypeVar

import pytest
import numpy as np
import sympy as sp

import mappings

from pyccel import lambdify as pyc_lambdify
from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

RTOL = 1e-13
ATOL = 1e-14

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [f for f in files if f.endswith(".py")]
T = TypeVar('T', 'float[:]', 'float[:,:]')

@pytest.mark.parametrize( "f", files )
@pytest.mark.skip(reason="Broken symbolic function support, see issue #330")
def test_symbolic(f, language):

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name, language)
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

        assert np.allclose(sp_x(r, p), pyc_x(r, p), rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_y(r, p), pyc_y(r, p), rtol=RTOL, atol=ATOL)

        pyc_x = pyc_lambdify(expr_x, {x : 'T', y : 'T'}, context_dict = {'T': T},
                    language = language)
        pyc_y = pyc_lambdify(expr_y, {x : 'T', y : 'T'}, context_dict = {'T': T},
                    language = language)

        assert np.allclose(sp_x(r, p), pyc_x(r, p), rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_y(r, p), pyc_y(r, p), rtol=RTOL, atol=ATOL)

        assert np.allclose(sp_x(r[0,:], p[0,:]), pyc_x(r[0,:], p[0,:]), rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_y(r[0,:], p[0,:]), pyc_y(r[0,:], p[0,:]), rtol=RTOL, atol=ATOL)

def test_lambdify_out_arg(language):
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
                    use_out = True, language = language)
        pyc_y = pyc_lambdify(expr_y, {x : 'float[:,:]', y : 'float[:,:]'}, result_type = 'float[:,:]',
                    use_out = True, language = language)

        print(pyc_x.__doc__)

        sp_out_x = np.empty_like(r)
        sp_out_y = np.empty_like(r)
        pyc_out_x = np.empty_like(r)
        pyc_out_y = np.empty_like(r)
        sp_out_x = sp_x(r, p)
        sp_out_y = sp_y(r, p)
        pyc_x(r, p, pyc_out_x)
        pyc_y(r, p, out = pyc_out_y)

        assert np.allclose(sp_out_x, pyc_out_x, rtol=RTOL, atol=ATOL)
        assert np.allclose(sp_out_y, pyc_out_y, rtol=RTOL, atol=ATOL)

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
