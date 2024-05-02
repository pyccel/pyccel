# pylint: disable=missing-function-docstring, missing-module-docstring
import os

from pyccel.parser.parser   import Parser
from pyccel.errors.errors   import Errors

from pyccel.ast.basic       import Basic
from pyccel.ast.core        import Assign, Return, FunctionDef
from pyccel.ast.literals    import LiteralInteger
from pyccel.ast.operators   import PyccelOperator, PyccelAdd, PyccelMinus
from pyccel.ast.variable    import Variable

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

def get_functions(filename):
    pyccel = Parser(filename)
    errors = Errors()

    ast = pyccel.parse()

    # Assert syntactic success
    assert(not errors.has_errors())

    settings = {}
    ast = pyccel.annotate(**settings)

    # Assert semantic success
    assert(not errors.has_errors())

    return list(ast.scope.functions.values())

def test_get_attribute_nodes():
    filename = os.path.join(path_dir, "math.py")

    fst = get_functions(filename)[0]
    atts = fst.get_attribute_nodes(Variable)

    assert(all(isinstance(a, Variable) for a in atts))

    expected = [Variable('int', 'a'),
                Variable('int', 'b'),
                Variable('int', 'c'),
                Variable('int', 'd'),
                Variable('int', 'e'),
                Variable('int', 'g')]

    for e in expected:
        assert(e in atts)

def test_get_attribute_nodes_exclude():
    filename = os.path.join(path_dir, "math.py")

    fst = get_functions(filename)[0]
    atts = fst.get_attribute_nodes(PyccelOperator, excluded_nodes=(PyccelAdd,))

    assert(all(isinstance(a, PyccelOperator) for a in atts))
    assert(all(not isinstance(a, PyccelAdd) for a in atts))
    assert(len(atts)==1)
    minus = atts[0]
    assert(isinstance(minus, PyccelMinus)==1)

    a = Variable('int', 'a')
    b = Variable('int', 'b')
    assert(minus.args[0] == a)
    assert(minus.args[1] == b)

def test_get_user_nodes():
    filename = os.path.join(path_dir, "math.py")

    interesting_var = Variable('int', 'a')

    fst = get_functions(filename)[0]
    atts = set(fst.get_attribute_nodes(Variable))
    atts = [v for v in atts  if v == interesting_var]

    a_var = atts[0]

    sums = set(a_var.get_user_nodes((PyccelAdd, PyccelMinus)))

    assert(all(isinstance(s, (PyccelAdd, PyccelMinus)) for s in sums))

    assert(len(sums) == 3)

def test_get_user_nodes_excluded():
    filename = os.path.join(path_dir, "math.py")

    interesting_var = Variable('int', 'a')

    fst = get_functions(filename)[0]
    atts = set(fst.get_attribute_nodes(Variable))
    atts = [v for v in atts  if v == interesting_var]

    a_var = atts[0]

    plus_assign = a_var.get_user_nodes(Assign, excluded_nodes = PyccelMinus)
    assert(len(plus_assign)==2)

def test_get_direct_user_nodes():
    filename = os.path.join(path_dir, "math.py")

    interesting_var = Variable('int', 'a')

    fst = get_functions(filename)[0]
    atts = set(fst.get_attribute_nodes(Variable))
    atts = [v for v in atts  if v == interesting_var]

    a_var = atts[0]

    sums = set(a_var.get_direct_user_nodes(lambda x : isinstance(x,(PyccelAdd, PyccelMinus))))

    assert(all(isinstance(s, (PyccelAdd, PyccelMinus)) for s in sums))

    assert(len(sums) == 2)

def test_substitute():
    filename = os.path.join(path_dir, "math.py")

    interesting_var = Variable('int', 'a')
    new_var = Variable('int', 'Z')

    fst = get_functions(filename)[0]
    atts = set(fst.get_attribute_nodes(Variable))
    atts = [v for v in atts  if v == interesting_var]

    a_var = atts[0]
    old_parents = a_var.get_user_nodes(Basic)
    assert(len(a_var.get_user_nodes(Basic))>0)

    fst.substitute(a_var, new_var)

    assert(len(a_var.get_user_nodes(Basic))==0)

    atts = set(fst.get_attribute_nodes(Variable))
    assert(new_var in atts)
    assert(set(new_var.get_user_nodes(Basic)) == set(old_parents))

def test_substitute_exclude():
    filename = os.path.join(path_dir, "math.py")

    interesting_var = Variable('int', 'a')
    new_var = Variable('int', 'Z')

    fst = get_functions(filename)[0]
    atts = set(fst.get_attribute_nodes(Variable))
    atts = [v for v in atts  if v == interesting_var]

    a_var = atts[0]
    old_parents = set(a_var.get_user_nodes(Basic))
    assert(len(a_var.get_user_nodes(Basic))>0)

    fst.substitute(a_var, new_var, excluded_nodes=(PyccelMinus))

    assert(len(a_var.get_user_nodes(Basic))==1)

    atts = set(fst.get_attribute_nodes(Variable))
    assert(new_var in atts)

    new_parents = set(new_var.get_user_nodes(Basic))
    assert(len(new_parents.difference(old_parents))==0)
    assert(len(old_parents.difference(new_parents))==1)

def test_recursive():
    filename = os.path.join(path_dir, "cyclic_dependence.py")

    fst = get_functions(filename)[0]

    atts = fst.get_attribute_nodes(PyccelMinus)
    assert(len(set(atts))==2)

    var = atts[0].args[0]

    atts = var.get_user_nodes(FunctionDef)
    assert(len(set(atts))==2) # The actual FunctionDef + the signature version

    fst.substitute(LiteralInteger(2), LiteralInteger(3))

