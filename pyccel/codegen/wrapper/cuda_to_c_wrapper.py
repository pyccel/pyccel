# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CudaToPythonWrapper
which creates an interface exposing Cuda code to C.
"""

from pyccel.ast.bind_c      import BindCModule
from pyccel.errors.errors   import Errors
from pyccel.ast.bind_c      import BindCVariable
from .wrapper               import Wrapper

errors = Errors()

class CudaToCWrapper(Wrapper):
    """
    Class for creating a wrapper exposing Cuda code to C.

    While CUDA is typically compatible with C by default.
    this wrapper becomes necessary in scenarios where specific adaptations
    or modifications are required to ensure seamless integration with C.
    """

    def _wrap_Module(self, expr):
        """
        Create a Module which is compatible with C.

        Create a Module which provides an interface between C and the
        Module described by expr.

        Parameters
        ----------
        expr : pyccel.ast.core.Module
            The module to be wrapped.

        Returns
        -------
        pyccel.ast.core.BindCModule
            The C-compatible module.
        """
        init_func = expr.init_func
        if expr.interfaces:
            errors.report("Interface wrapping is not yet supported for Cuda",
                      severity='warning', symbol=expr)
        if expr.classes:
            errors.report("Class wrapping is not yet supported for Cuda",
                      severity='warning', symbol=expr)

        variables = [self._wrap(v) for v in expr.variables]

        return BindCModule(expr.name, variables, expr.funcs,
                init_func=init_func,
                scope = expr.scope,
                original_module=expr)

    def _wrap_Variable(self, expr):
        """
        Create all objects necessary to expose a module variable to C.

        Create and return the objects which must be printed in the wrapping
        module in order to expose the variable to C

        Parameters
        ----------
        expr : pyccel.ast.variables.Variable
            The module variable.

        Returns
        -------
        pyccel.ast.core.BindCVariable
            The C-compatible variable. which must be printed in
            the wrapping module to expose the variable.
        """
        return expr.clone(expr.name, new_class = BindCVariable)

