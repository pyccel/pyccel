""" A script to provide a hook which allows artifacts to be generated during installation of the package.
"""
import os
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    """
    A class which allows code to be run while building the package.

    A class inheriting from BuildHookInterface which allows code to be run to create artifacts during the build stage
    of the package installation using hatch.
    See <https://hatch.pypa.io/latest/plugins/build-hook/reference> for more details.
    """

    def initialize(self, version, build_data):
        """
        Script run before creating the build target.

        See <https://hatch.pypa.io/latest/plugins/build-hook/reference/#hatchling.builders.hooks.plugin.interface.BuildHookInterface.initialize>.

        Parameters
        ----------
        version : str
            See hatch documentation.

        build_data : dict
            See hatch documentation.
        """
        folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','pyccel','stdlib','internal'))
        files_stubs = ['blas', 'dfftpack', 'fitpack',
                'lapack', 'mpi', 'openacc', 'openmp']
        output_files = [os.path.join(folder, f)+'.pyccel' for f in files_stubs]
        for f in output_files:
            if os.path.isfile(f):
                os.remove(f)

        subprocess.run([sys.executable, "-c", "from pyccel.commands.pyccel_init import pyccel_init; pyccel_init()"],
                cwd = self.root, check=True)
        build_data['artifacts'].extend(output_files)
