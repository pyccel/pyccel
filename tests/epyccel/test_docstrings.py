# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import docstrings

from pyccel import epyccel

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_docstrings_mod(language):
    return epyccel_module_with_fallback(docstrings, language)


def pad_docstrings(python_doc, pyccel_doc):
    # Remove empty lines as ast does not preserve them
    python_doc = [p for p in python_doc.split("\n") if p.strip()]
    pyccel_doc = [p for p in pyccel_doc.split("\n") if p.strip()]

    # Pad the smaller doc string to ensure a match
    extra_spaces = len(python_doc[0]) - len(pyccel_doc[0])
    if extra_spaces > 0:
        pyccel_doc = [" " * extra_spaces + p for p in pyccel_doc]
    if extra_spaces < 0:
        extra_spaces = -extra_spaces
        python_doc = [" " * extra_spaces + p for p in python_doc]

    python_doc = "\n".join(python_doc)
    pyccel_doc = "\n".join(pyccel_doc)
    return python_doc, pyccel_doc


def test_1_line_docstring(epyc_docstrings_mod):
    f = docstrings.n1_line_docstring
    g = epyc_docstrings_mod.n1_line_docstring
    assert f.__doc__.strip() == g.__doc__.strip()


def test_multiline_line_docstring(epyc_docstrings_mod):
    f = docstrings.multiline_line_docstring
    g = epyc_docstrings_mod.multiline_line_docstring

    python_doc, pyccel_doc = pad_docstrings(f.__doc__, g.__doc__)

    assert python_doc == pyccel_doc


def test_class_docstring(epyc_docstrings_mod):
    A = docstrings.MyClass
    B = epyc_docstrings_mod.MyClass

    python_doc, pyccel_doc = pad_docstrings(A.__doc__, B.__doc__)
    assert python_doc == pyccel_doc


def test_property_docstring(epyc_docstrings_mod):
    MyA = docstrings.MyClassProperty
    B = epyc_docstrings_mod.MyClassProperty

    print(MyA.__doc__, B.__doc__)

    python_doc, pyccel_doc = pad_docstrings(MyA.__doc__, B.__doc__)
    assert python_doc == pyccel_doc
    python_doc, pyccel_doc = pad_docstrings(MyA.x.__doc__, B.x.__doc__)
    assert python_doc == pyccel_doc


def test_module_docstring(experimental_language):
    from modules import Module_docstring as mod

    epyc_mod = epyccel(mod, language=experimental_language)
    python_doc, pyccel_doc = pad_docstrings(mod.__doc__, epyc_mod.__doc__)
    assert python_doc == pyccel_doc
