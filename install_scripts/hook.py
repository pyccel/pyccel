""" A script to provide a hook which allows artifacts to be generated during installation of the package.
"""
from pathlib import Path
import shutil
import subprocess

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
    **kwargs : dict
        See hatch docs.
    """

    def __init__(self, *args, **kwargs): #pylint: disable=useless-parent-delegation
        super().__init__(*args, **kwargs)

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
        pyccel_root = Path(__file__).parent.parent
        subprocess.run([shutil.which('git'), 'submodule', 'update', '--init'], cwd = pyccel_root, check=True)
        gFTL_folder = (pyccel_root / 'pyccel' / 'extensions' / 'gFTL').absolute()
        subprocess.run([shutil.which('git'), 'clean', '-fd'], cwd = gFTL_folder, check=True)
        shutil.rmtree(gFTL_folder / 'build', ignore_errors = True)
        subprocess.run([shutil.which('cmake'), '-S', str(gFTL_folder), '-B', str(gFTL_folder / 'build'),
                        f'-DCMAKE_INSTALL_PREFIX={gFTL_folder / "install"}'], cwd = gFTL_folder, check=True)
        subprocess.run([shutil.which('cmake'), '--build', str(gFTL_folder / 'build')], cwd = gFTL_folder, check=True)
        subprocess.run([shutil.which('cmake'), '--install', str(gFTL_folder / 'build')], cwd = gFTL_folder, check=True)

        build_data['artifacts'].extend((gFTL_folder / 'install' / 'GFTL-1.13' / 'include').glob('**/*'))

