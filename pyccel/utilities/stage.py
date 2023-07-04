#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" File containing information about which treatment stage Pyccel is executing
"""
from .metaclasses import Singleton

class PyccelStage(metaclass = Singleton):
    """ Class wrapping a string indicating which treatment stage Pyccel is executing.
    This string is one of:
     - syntactic
     - semantic
     - codegen
     - cwrapper

    When Pyccel is not executing the stage is None
    """
    def __init__(self):
        self._stage = None

    def set_stage(self, stage):
        """ Set the current treatment stage
        """
        assert stage in ('syntactic', 'semantic', 'codegen', 'cwrapper')
        self._stage = stage

    def __eq__(self, other):
        return self._stage == other

    def pyccel_finished(self):
        """ Indicate that Pyccel has finished running and reset stage to None
        """
        self._stage = None

    @property
    def current_stage(self):
        """
        Get the current stage as a string.

        Returns one of:
        - syntactic
        - semantic
        - codegen
        - cwrapper
        indicating the current stage.
        """
        return self._stage
