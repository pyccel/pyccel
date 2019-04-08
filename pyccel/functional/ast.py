# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy
from sympy import Tuple, IndexedBase
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Integer, Float
from sympy import sympify
from sympy import FunctionClass


from pyccel.codegen.utilities import random_string
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.datatypes import Int, Real, Complex, Bool
from pyccel.ast.core import Slice
from pyccel.ast.core import Variable, FunctionDef, Assign, AugAssign
from pyccel.ast.core import Return
from .core import Reduce
from .core import SeqMap, ParMap, BasicMap
from .core import SeqTensorMap, ParTensorMap, BasicTensorMap
from .core import SeqZip, SeqProduct
from .core import ParZip, ParProduct
from .core import assign_type, BasicTypeVariable
from .core import TypeVariable, TypeTuple, TypeList
from .core import VariableGenerator
from .core import generator_as_block
from .semantic import Parser as SemanticParser

from .semantic import _known_functions
from .semantic import _functors_map_registery
from .semantic import _functors_registery
from .semantic import _elemental_math_functions
from .semantic import _math_vector_functions
from .semantic import _math_matrix_functions
from .semantic import _math_functions
from .semantic import _known_functions
from .semantic import _known_functions
from .semantic import _known_functions



#==============================================================================
# ...
def _attributs_from_type(t, d_var):
    if isinstance(t, TypeList):
        t = _attributs_from_type(t.parent, d_var)
        d_var['rank'] = d_var['rank'] + 1
        return t, d_var

    elif isinstance(t, TypeTuple):
        raise NotImplementedError()

    elif isinstance(t, TypeVariable):
        d_var['dtype']          = t.dtype
        d_var['rank']           = t.rank
        d_var['is_stack_array'] = t.is_stack_array
        d_var['order']          = t.order
        d_var['precision']      = t.precision

        return t, d_var
# ...

# ... default values
def _attributs_default():
    d_var = {}

    d_var['dtype']          = None
    d_var['rank']           = 0
    d_var['allocatable']    = False
    d_var['is_stack_array'] = False
    d_var['is_pointer']     = False
    d_var['is_target']      = False
    d_var['shape']          = None
    d_var['order']          = 'C'
    d_var['precision']      = None

    return d_var
# ...
#==============================================================================

#==============================================================================
class AST(object):

    def __init__(self, parser, **kwargs):
        assert(isinstance(parser, SemanticParser))

        # ...
        self._expr            = parser.expr
        self._namespace       = parser.namespace
        self._d_types         = parser.d_types
        self._d_expr          = parser.d_expr
        self._tag             = parser.tag
        self.main             = parser.main
        self.main_type        = parser.main_type
        self._typed_functions = parser.typed_functions
        self._default_type    = parser.default_type
        # ...

    @property
    def expr(self):
        return self._expr

    @property
    def typed_functions(self):
        return self._typed_functions

    @property
    def default_type(self):
        return self._default_type

    @property
    def namespace(self):
        return self._namespace

    @property
    def d_types(self):
        return self._d_types

    @property
    def d_expr(self):
        return self._d_expr

    @property
    def tag(self):
        return self._tag

    def inspect(self):
        print('============ types =============')
        print(self.d_types)
        for k,v in self.d_types.items():
            print('  {k} = {v}'.format(k=k, v=v.view()))
        print('================================')

    def _get_label(self, target, domain=False, codomain=False):
        # TODO improve
        if codomain:
            assert(not domain)
            if (isinstance(target, FunctionClass)):
                name = str(target)

            else:
                name = str(target.name)

            return name

        if domain:
            assert(not codomain)
            if (isinstance(target, FunctionClass)):
                name = str(target)

            else:
                name = str(target.name)

            _avail_funcs = list(self.typed_functions.keys()) + _math_functions
            if name in _avail_funcs:
                return name + '_args'

        if isinstance(target, FunctionDef):
            return str(target.name) + '_args'

        elif isinstance(target, UndefinedFunction):
            return str(target)

        elif isinstance(target, Symbol):
            return target.name

        else:
            raise NotImplementedError('for {}'.format(type(target)))

    def _get_type(self, target, domain=False, codomain=False):
        label = self._get_label(target, domain=domain, codomain=codomain)

        if label in self.d_types.keys():
            return self.d_types[label]

        return None

    def _set_type(self, target, value, domain=False, codomain=False):
        label = self._get_label(target, domain=domain, codomain=codomain)

        self.d_types[label] = value
        self._set_expr(value, target)

    def _set_expr(self, t_var, expr):
        self._d_expr[t_var.name] = expr

    def build_namespace(self):
        """builds the namespace from types."""
        raise NotImplementedError('')

    def annotate(self, stmt=None):

        if stmt is None:
            stmt = self.expr

        cls = type(stmt)
        name = cls.__name__

        method = '_annotate_{}'.format(name)
        if hasattr(self, method):
            return getattr(self, method)(stmt)

        elif name in _known_functions.keys():
            if name == 'map':
                func, target = stmt.args

                # ... construct the generator
                # TODO compute its depth from type of target
                depth     = None

                generator = self.annotate(target)
                if isinstance(generator, Variable):
                    generator = VariableGenerator(generator)
                # ...

                # ... construct the results
                type_codomain = self.main_type
                results = self.annotate(type_codomain)

                # compute depth of the type list
                depth_out = len(list(type_codomain.atoms(TypeList)))
                # ...

                # ... apply the function to arguments
                index    = generator.index
                iterator = generator.iterator

                if isinstance(iterator, Tuple):
                    rhs = func( *iterator )

                else:
                    rhs = func( iterator )
                # ...

                # ... create lhs
                lhs = generator.iterator
                # TODO check this
                if isinstance(lhs, Tuple) and len(lhs) == 1:
                    lhs = lhs[0]
                # ...

                # ... create lhs for storing the result
                if isinstance(results, Variable):
                    results = [results]

                else:
                    raise NotImplementedError()

                if not isinstance(index, Tuple):
                    index = [index]

                else:
                    index = list([i for i in index])

                lhs = []
                for r in results:
                    m = r.rank - depth_out
                    ind = index + [Slice(None, None)] * m
                    if len(ind) == 1:
                        ind = ind[0]

                    lhs.append(IndexedBase(r.name)[ind])

                lhs = Tuple(*lhs)
                if len(lhs) == 1:
                    lhs = lhs[0]
                # ...

                # ... create core statement
                stmts = [Assign(lhs, rhs)]
                # ...

                # TODO USE THIS
#                expr = self.get_expr_from_type()

                # return the associated for loops
                return generator_as_block( generator, stmts,
                                           parallel      = False )

            else:
                raise NotImplementedError('')

        # Unknown object, we raise an error.
        raise TypeError('{node} not yet available'.format(node=type(stmt)))

    def _annotate_Lambda(self, stmt):
        args = [self.annotate(i) for i in stmt.variables]
        expr = self.annotate(stmt.expr)
        # TODO improve
        results = self.annotate(self.main)
        if not isinstance(results, (list, tuple, Tuple)):
            results = [results]

        # TODO improve
        body = [expr]

        if len(results) == 1:
            body += [Return(results[0])]

        else:
            body += [Return(results)]

        # ...
        decorators = {'types':         build_types_decorator(args),
                      'external_call': []}

        tag         = random_string( 6 )
        name      = 'lambda_{}'.format( tag )
        # ...

        return FunctionDef(name, args, results, body,
                           decorators=decorators)

        return expr

    def _annotate_Symbol(self, stmt):
        t_var = self.d_types[stmt.name]
        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, stmt.name, **d_var )

        return var

    def _annotate_Integer(self, stmt):
        return stmt

    def _annotate_Float(self, stmt):
        return stmt

    def _annotate_TypeVariable(self, stmt):
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def _annotate_TypeTuple(self, stmt):
        # TODO
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def _annotate_TypeList(self, stmt):
        # TODO
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def get_expr_from_type(self, t_var=None):
        if t_var is None:
            t_var = self.main_type

        return self.d_expr[t_var.name]
