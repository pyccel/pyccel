# coding: utf-8

import numpy as np


from pyccel.types.ast import DataTypeFactory
from pyccel.types.ast import FunctionDef, ClassDef
from pyccel.types.ast import Variable, DottedName

# TODO: check arguments

##########################################################
#                     DNS matrix
##########################################################
class Matrix_dns_create(FunctionDef):
    """ Represents a Matrix create procedure. """
    def __new__(cls):
        """
        Represents a call to create for a dns matrix.

        Matrix_dns_create is implemented as a FunctionDef, where the result is
        an instance of Matrix_dns. This is done by specifying the result of the
        create using the DataTypeFactory.
        """
        # ...
        name = 'create'

        cls._name = name
        # ...

        # ...
        n_rows = Variable('int', 'n_rows')
        n_cols = Variable('int', 'n_cols')
        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')
        # ...

        # ...
        args = [n_rows, n_cols, n_block_rows, n_block_cols]
        # ...

        # ...
        dtype = DataTypeFactory('Matrix_dns', ("_name"))()
        var_name = 'var_%d' % abs(hash(name))
        var      = Variable(dtype, var_name)

        results     = [var]
        # ...

        # ...
        body        = []
        local_vars  = []
        global_vars = []
        hide        = False
        kind        = 'procedure'
        cls_name    = '__UNDEFINED__'
        # ...

        return FunctionDef.__new__(cls, name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, kind=kind, cls_name=cls_name)

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint

        name    = 'Matrix_dns_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

class Matrix_dns(ClassDef):
    """
    Represents a DNS Matrix from plaf.

    Examples

    >>> from pyccel.clapp.plaf import Matrix_dns
    >>> M = Matrix_dns()
    >>> M.methods
    (<class 'pyccel.clapp.plaf.matrix.Matrix_dns_create'>,)
    """
    _instance = 'dns'

    def __new__(cls):
        # ...
        n_rows       = Variable('int', 'n_rows')
        n_cols       = Variable('int', 'n_cols')
        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')

        # TODO defined as arr_a or arr_c if using complex numbers
        _data = 'arr_a'
        data = Variable('double', _data, \
                        rank=1, shape=(n_rows, n_cols), \
                        allocatable=True)
        # ...

        # ...
        attributs = [n_rows, n_cols, n_block_rows, n_block_cols, data]
        methods   = [Matrix_dns_create]

        options     = ['public']
        # ...

        return ClassDef.__new__(cls, 'matrix_dns', \
                                attributs, methods, \
                                options=options)

    @property
    def module(self):
        return 'plf_m_matrix_{0}'.format(self._instance)

    @property
    def dtype(self):
        return 'plf_t_matrix_{0}'.format(self._instance)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{}'.format(sstr(self.name))
##########################################################
