# coding: utf-8

import numpy as np


from pyccel.ast.core import DataTypeFactory
from pyccel.ast.core import FunctionDef, ClassDef
from pyccel.ast.core import Variable, DottedName

# TODO: - check arguments
#       - sets the right shape and rank for array attributs

##########################################################
#                     dns matrix
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
        var      = Variable(dtype, 'self')

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
    Represents a dns Matrix from plaf.

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
        arr_a = Variable('double', 'arr_a', \
                        rank=1, shape=(n_rows, n_cols), \
                        allocatable=True)
        # ...

        # ...
        attributs = [n_rows, n_cols, n_block_rows, n_block_cols, arr_a]
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

##########################################################
#                     csr matrix
##########################################################
class Matrix_csr_create(FunctionDef):
    """ Represents a Matrix create procedure. """
    def __new__(cls):
        """
        Represents a call to create for a csr matrix.

        Matrix_csr_create is implemented as a FunctionDef, where the result is
        an instance of Matrix_csr. This is done by specifying the result of the
        create using the DataTypeFactory.
        """
        # ...
        name = 'create'

        cls._name = name
        # ...

        # ...
        n_rows = Variable('int', 'n_rows')
        n_cols = Variable('int', 'n_cols')
        n_nnz  = Variable('int', 'n_nnz')
        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')
        # ...

        # ...
        args = [n_rows, n_cols, n_nnz, n_block_rows, n_block_cols]
        # ...

        # ...
        dtype = DataTypeFactory('Matrix_csr', ("_name"))()
        var = Variable(dtype, 'self')

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

        name    = 'Matrix_csr_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

class Matrix_csr(ClassDef):
    """
    Represents a csr Matrix from plaf.

    Examples

    >>> from pyccel.clapp.plaf import Matrix_csr
    >>> M = Matrix_csr()
    >>> M.methods
    (<class 'pyccel.clapp.plaf.matrix.Matrix_csr_create'>,)
    """
    _instance = 'csr'

    def __new__(cls):
        # ...
        n_rows       = Variable('int', 'n_rows')
        n_cols       = Variable('int', 'n_cols')
        n_nnz        = Variable('int', 'n_nnz')
        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')

        # TODO defined as arr_a or arr_c if using complex numbers
        arr_a = Variable('double', 'arr_a', \
                        rank=2, shape=(n_rows, n_cols), \
                        allocatable=True)

        arr_ia = Variable('int', 'arr_ia', \
                        rank=1, shape=n_nnz, \
                        allocatable=True)

        arr_ja = Variable('int', 'arr_ja', \
                        rank=1, shape=n_nnz, \
                        allocatable=True)
        # ...

        # ...
        attributs = [n_rows, n_cols, n_nnz, n_block_rows, n_block_cols, \
                     arr_a, arr_ia, arr_ja]
        methods   = [Matrix_csr_create()]

        options     = ['public']
        # ...

        return ClassDef.__new__(cls, 'matrix_csr', \
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

##########################################################
#                     csc matrix
##########################################################
class Matrix_csc_create(FunctionDef):
    """ Represents a Matrix create procedure. """
    def __new__(cls):
        """
        Represents a call to create for a csc matrix.

        Matrix_csc_create is implemented as a FunctionDef, where the result is
        an instance of Matrix_csc. This is done by specifying the result of the
        create using the DataTypeFactory.
        """
        # ...
        name = 'create'

        cls._name = name
        # ...

        # ...
        n_rows = Variable('int', 'n_rows')
        n_cols = Variable('int', 'n_cols')
        n_nnz  = Variable('int', 'n_nnz')
        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')
        # ...

        # ...
        args = [n_rows, n_cols, n_nnz, n_block_rows, n_block_cols]
        # ...

        # ...
        dtype = DataTypeFactory('Matrix_csc', ("_name"))()
        var      = Variable(dtype, 'self')

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

        name    = 'Matrix_csc_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

class Matrix_csc(ClassDef):
    """
    Represents a csc Matrix from plaf.

    Examples

    >>> from pyccel.clapp.plaf import Matrix_csc
    >>> M = Matrix_csc()
    >>> M.methods
    (<class 'pyccel.clapp.plaf.matrix.Matrix_csc_create'>,)
    """
    _instance = 'csc'

    def __new__(cls):
        # ...
        n_rows       = Variable('int', 'n_rows')
        n_cols       = Variable('int', 'n_cols')
        n_nnz        = Variable('int', 'n_nnz')
        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')

        # TODO defined as arr_a or arr_c if using complex numbers
        arr_a = Variable('double', 'arr_a', \
                        rank=2, shape=(n_rows, n_cols), \
                        allocatable=True)

        arr_ia = Variable('int', 'arr_ia', \
                        rank=1, shape=n_nnz, \
                        allocatable=True)

        arr_ja = Variable('int', 'arr_ja', \
                        rank=1, shape=n_nnz, \
                        allocatable=True)
        # ...

        # ...
        attributs = [n_rows, n_cols, n_nnz, n_block_rows, n_block_cols, \
                     arr_a, arr_ia, arr_ja]
        methods   = [Matrix_csc_create]

        options     = ['public']
        # ...

        return ClassDef.__new__(cls, 'matrix_csc', \
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

##########################################################
#                     bnd matrix
##########################################################
class Matrix_bnd_create(FunctionDef):
    """ Represents a Matrix create procedure. """
    def __new__(cls):
        """
        Represents a call to create for a bnd matrix.

        Matrix_bnd_create is implemented as a FunctionDef, where the result is
        an instance of Matrix_bnd. This is done by specifying the result of the
        create using the DataTypeFactory.
        """
        # ...
        name = 'create'

        cls._name = name
        # ...

        # ...
        n_rows = Variable('int', 'n_rows')
        n_cols = Variable('int', 'n_cols')
        n_ml   = Variable('int', 'n_ml')
        n_mu   = Variable('int', 'n_mu')
        n_abd  = Variable('int', 'n_abd')
        n_lowd = Variable('int', 'n_lowd')

        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')
        # ...

        # ...
        args = [n_ml, n_mu, n_abd, n_lowd, \
                n_rows, n_cols, n_block_rows, n_block_cols]
        # ...

        # ...
        dtype = DataTypeFactory('Matrix_bnd', ("_name"))()
        var = Variable(dtype, 'self')

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

        name    = 'Matrix_bnd_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

class Matrix_bnd(ClassDef):
    """
    Represents a bnd Matrix from plaf.

    Examples

    >>> from pyccel.clapp.plaf import Matrix_bnd
    >>> M = Matrix_bnd()
    >>> M.methods
    (<class 'pyccel.clapp.plaf.matrix.Matrix_bnd_create'>,)
    """
    _instance = 'bnd'

    def __new__(cls):
        # ...
        n_rows       = Variable('int', 'n_rows')
        n_cols       = Variable('int', 'n_cols')
        n_ml         = Variable('int', 'n_ml')
        n_mu         = Variable('int', 'n_mu')
        n_abd        = Variable('int', 'n_abd')
        n_lowd       = Variable('int', 'n_lowd')

        n_block_rows = Variable('int', 'n_block_rows')
        n_block_cols = Variable('int', 'n_block_cols')

        # TODO defined as arr_a or arr_c if using complex numbers
        arr_abd = Variable('double', 'arr_abd', \
                        rank=1, shape=(n_rows, n_cols), \
                        allocatable=True)
        # ...

        # ...
        attributs = [n_ml, n_mu, n_abd, n_lowd, \
                     n_rows, n_cols, n_block_rows, n_block_cols, arr_abd]
        methods   = [Matrix_bnd_create]

        options     = ['public']
        # ...

        return ClassDef.__new__(cls, 'matrix_bnd', \
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

##########################################################
#                     coo matrix
##########################################################
class Matrix_coo_create(FunctionDef):
    """ Represents a Matrix create procedure. """
    def __new__(cls):
        """
        Represents a call to create for a coo matrix.

        Matrix_coo_create is implemented as a FunctionDef, where the result is
        an instance of Matrix_coo. This is done by specifying the result of the
        create using the DataTypeFactory.
        """
        # ...
        name = 'create'

        cls._name = name
        # ...

        # ...
        n_rows = Variable('int', 'n_rows')
        n_cols = Variable('int', 'n_cols')
        n_nnz  = Variable('int', 'n_nnz')
        # ...

        # ...
        args = [n_rows, n_cols, n_nnz]
        # ...

        # ...
        dtype = DataTypeFactory('Matrix_coo', ("_name"))()
        var = Variable(dtype, 'self')

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

        name    = 'Matrix_coo_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

class Matrix_coo(ClassDef):
    """
    Represents a coo Matrix from plaf.

    Examples

    >>> from pyccel.clapp.plaf import Matrix_coo
    >>> M = Matrix_coo()
    >>> M.methods
    (<class 'pyccel.clapp.plaf.matrix.Matrix_coo_create'>,)
    """
    _instance = 'coo'

    def __new__(cls):
        # ...
        n_rows = Variable('int', 'n_rows')
        n_cols = Variable('int', 'n_cols')
        n_nnz  = Variable('int', 'n_nnz')

        # TODO defined as arr_a or arr_c if using complex numbers
        arr_a = Variable('double', 'arr_a', \
                        rank=1, shape=(n_rows, n_cols), \
                        allocatable=True)

        arr_ia = Variable('int', 'arr_ia', \
                        rank=1, shape=n_nnz, \
                        allocatable=True)

        arr_ja = Variable('int', 'arr_ja', \
                        rank=1, shape=n_nnz, \
                        allocatable=True)
        # ...

        # ...
        attributs = [n_rows, n_cols, n_nnz, \
                     arr_a, arr_ia, arr_ja]
        methods   = [Matrix_coo_create]

        options     = ['public']
        # ...

        return ClassDef.__new__(cls, 'matrix_coo', \
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
