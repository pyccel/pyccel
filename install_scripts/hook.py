""" A script to provide a hook which allows artifacts to be generated during installation of the package.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    """
    A class which allows code to be run while building the package.

    A class inheriting from BuildHookInterface which allows code to be run to create artifacts during the build stage
    of the package installation using hatch.
    See <https://hatch.pypa.io/latest/plugins/build-hook/reference> for more details.

    Parameters
    ----------
    *args : tuple
        See hatch docs.
    **kwds : dict
        See hatch docs.
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
        # Build gFTL for installation
        gFTL_folder = (Path(__file__).parent.parent / 'pyccel' / 'extensions' / 'gFTL').absolute()
        subprocess.run([shutil.which('cmake'), '-S', str(gFTL_folder), '-B', str(gFTL_folder / 'build'),
                        f'-DCMAKE_INSTALL_PREFIX={gFTL_folder / "install"}'], cwd = gFTL_folder, check=True)
        subprocess.run([shutil.which('cmake'), '--build', str(gFTL_folder / 'build')], cwd = gFTL_folder, check=True)
        subprocess.run([shutil.which('cmake'), '--install', str(gFTL_folder / 'build')], cwd = gFTL_folder, check=True)

        build_data['artifacts'].extend((gFTL_folder / 'install' / 'GFTL-1.13' / 'include').glob('**/*'))

        # Build pickle files
        pickle_folder = (Path(__file__).parent.parent / 'pyccel' / 'stdlib' / 'internal').absolute()
        files_stubs = ['blas', 'dfftpack', 'fitpack',
                'lapack', 'mpi', 'openacc', 'openmp']
        output_files = [pickle_folder / (f+'.pyccel') for f in files_stubs]
        for f in output_files:
            if f.is_file():
                os.remove(f)

        subprocess.run([sys.executable, "-c", "from pyccel.commands.pyccel_init import pyccel_init; pyccel_init()"],
                cwd = self.root, check=True)
        build_data['artifacts'].extend(output_files)
