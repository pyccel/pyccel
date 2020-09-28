import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import omp
from conftest       import *


def compare_epyccel(f, *args, language='fortran'):
    f2 = epyccel(f, language=language)
    out1 = f(*args)
    out2 = f2(*args)
    assert np.equal(out1, out2)
