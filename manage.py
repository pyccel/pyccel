# coding: utf-8

from pyccel.codegen.utilities import load_extension

####################################
if __name__ == '__main__':
    load_extension('math', 'extensions', silent=False)
    load_extension('math', 'extensions', modules=['bsplines'])
    load_extension('math', 'extensions', modules='quadratures')
