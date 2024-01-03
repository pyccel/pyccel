import os
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):

    def initialize(self, version, build_data):
        subprocess.run([sys.executable, "-c", "from pyccel.commands.pyccel_init import pyccel_init; pyccel_init()"],
                cwd = self.root, check=True)
        folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','pyccel','stdlib','internal'))
        files = ['blas', 'dfftpack', 'fitpack',
                'lapack', 'mpi', 'openacc', 'openmp']
        build_data['artifacts'].extend(os.path.join(folder, f'{f}.pyccel') for f in files)
