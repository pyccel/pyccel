import os

from pyccel.codegen.codegen import Codegen
from pyccel.parser.parser   import Parser
from pyccel.errors.errors   import Errors

from pyccel.ast.literals import LiteralInteger
from pyccel.ast.variable import Variable, ValuedVariable

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

def get_functions(filename):
    pyccel = Parser(filename)
    errors = Errors()

    ast = pyccel.parse()

    # Assert syntactic success
    assert(not errors.has_errors())
    print(ast, type(ast))

    settings = {}
    ast = pyccel.annotate(**settings)

    # Assert semantic success
    assert(not errors.has_errors())

    return list(ast.namespace.functions.values())

def test_get_attribute_nodes():
    filename = os.path.join(path_dir, "math.py")

    fst = get_functions(filename)[0]
    atts = fst.get_attribute_nodes(Variable)

    assert(all(isinstance(a, Variable) for a in atts))

    expected = [Variable('int', 'a'),
                Variable('int', 'b'),
                ValuedVariable('int', 'c'),
                Variable('int', 'd'),
                Variable('int', 'e'),
                Variable('int', 'f')]

    for e in expected:
        assert(e in atts)

def test_get_attribute_nodes_exclude():
    filename = os.path.join(path_dir, "math.py")

    fst = get_functions(filename)[0]
    atts = fst.get_attribute_nodes(Variable, excluded_nodes=(ValuedVariable,))

    assert(all(isinstance(a, Variable) for a in atts))

    expected = [Variable('int', 'a'),
                Variable('int', 'b'),
                Variable('int', 'd'),
                Variable('int', 'e'),
                Variable('int', 'f')]

    for e in expected:
        assert(e in atts)

    not_expected = ValuedVariable('int', 'c', value=LiteralInteger(0))
    assert(not_expected not in atts)
