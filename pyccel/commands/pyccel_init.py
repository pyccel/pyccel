# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing script to pyccelize internal files
"""
import os
from argparse import ArgumentParser
from pyccel.parser.parser import Parser

def pyccel_init():
    """ Pickle internal pyccel files
    """
    parser = ArgumentParser(description='Pickle internal pyccel files')
    parser.parse_args()

    folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','stdlib','internal'))
    files = ['blas.pyh', 'dfftpack.pyh', 'fitpack.pyh',
            'lapack.pyh', 'mpi.pyh', 'openacc.pyh', 'openmp.pyh']

    for f in files:
        parser = Parser(os.path.join(folder,f), show_traceback=False)
        parser.parse(verbose=False)
