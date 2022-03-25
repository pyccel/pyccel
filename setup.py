# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import setuptools
from setuptools.command.dist_info import dist_info

class PickleHeaders(dist_info):
    """ Class to pickle headers in time for them to be collected
    by the MANIFEST.in treatment
    """
    def run(self):

        # Just add a print for the example
        # Process .pyh headers and store their AST in .pyccel pickle files.
        import os
        from pyccel.parser.parser import Parser

        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyccel', 'stdlib', 'internal'))
        files = ['blas.pyh', 'dfftpack.pyh', 'fitpack.pyh',
                'lapack.pyh', 'mpi.pyh', 'openacc.pyh', 'openmp.pyh']

        for f in files:
            parser = Parser(os.path.join(folder, f), show_traceback=False)
            parser.parse(verbose=False)

        # Execute the classic dist_info command
        super().run()


if __name__ == "__main__":
    setuptools.setup(cmdclass={"dist_info": PickleHeaders})
