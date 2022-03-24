# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import setuptools
from setuptools.command.develop import develop

class PickleHeaders(develop):
    """ Class to handle post-install step which pickles headers
    """
    def run(self):
        # Execute the classic develop_data command
        super().run()

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


if __name__ == "__main__":
    setuptools.setup(cmdclass={"develop": PickleHeaders})
