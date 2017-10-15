# coding: utf-8

from pyccel.types.ast import DataTypeFactory, DottedName
from pyccel.types.ast import Assign
from pyccel.types.ast import Variable, FunctionDef, ClassDef, Module

# ...
def Point_definitions():
    c_name = 'Point'
    alias  = None
    c_dtype = DataTypeFactory(c_name, ("_name"), \
                              prefix='Custom', \
                              alias=alias)

    # ... classes
    this = Variable(c_dtype(), 'self')

    #     methods
    #     - __init__
    x = Variable('double', DottedName('self', 'x'))
    y = Variable('double', DottedName('self', 'y'))
    a = Variable('double', 'a')
    b = Variable('double', 'b')
    body = [Assign(x,a), Assign(y,b)]
    create = FunctionDef('__init__', [this, a, b], [], body)

    #     - translate
    x = Variable('double', DottedName('self', 'x'))
    y = Variable('double', DottedName('self', 'y'))
    a = Variable('double', 'a')
    b = Variable('double', 'b')
    body = [Assign(x,x+a), Assign(y,y+b)]
    translate = FunctionDef('translate', [this, a,b], [], body)

    # attributs 
    x = Variable('double', 'x')
    y = Variable('double', 'y')

    # class definition
    attributs   = [x,y]
    methods     = [create, translate]
    Point = ClassDef(c_name, attributs, methods)
    # ...

    # ... additional functions to the module
    x = Variable('double', 'x')
    y = Variable('double', 'y')
    incr = FunctionDef('incr', [x], [y], [Assign(y,x+1)])
    decr = FunctionDef('decr', [x], [y], [Assign(y,x-1)])
    # ...

    variables = []
#    funcs     = [incr, decr]
    funcs     = []
    classes   = [Point]

    d = {}
    d['modules'] = [Module('m_pyccel', variables, funcs, classes)]
    d['cls_constructs'] = [c_dtype()]
    d['classes'] = [Point]
#    d['funcs'] = [Point, incr, decr]

    return d
# ...

# ...
def stdlib_definitions():
    """Adds Pyccel stdlib functions, classes and constants to the namespace

    Returns

    namespace: dict
        dictionary containing all declared variables/functions/classes.

    declarations: dict
        dictionary containing all declarations.

    cls_constructs: dict
        dictionary of datatypes of classes using DatatypeFactory
    """
    namespace      = {}
    declarations   = {}
    cls_constructs = {}
    classes        = {}

    d = Point_definitions()

    stmts  = []
    stmts += d['modules']

    for i in d['cls_constructs']:
        cls_constructs[str(i.name)] = i

    for i in d['classes']:
        classes[str(i.name)] = i

    return namespace, declarations, cls_constructs, classes, stmts
# ...
