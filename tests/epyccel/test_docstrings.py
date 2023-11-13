# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.epyccel import epyccel

def pad_docstrings(python_doc, pyccel_doc):
    # Remove empty lines as ast does not preserve them
    python_doc = [p for p in python_doc.split('\n') if p.strip()]
    pyccel_doc = [p for p in pyccel_doc.split('\n') if p.strip()]

    # Pad the smaller doc string to ensure a match
    extra_spaces = len(python_doc[0]) - len(pyccel_doc[0])
    if extra_spaces>0:
        pyccel_doc = [' '*extra_spaces+p for p in pyccel_doc]
    if extra_spaces<0:
        extra_spaces = -extra_spaces
        python_doc = [' '*extra_spaces+p for p in python_doc]

    python_doc = '\n'.join(python_doc)
    pyccel_doc = '\n'.join(pyccel_doc)
    return python_doc, pyccel_doc

def test_1_line_docstring(language):
    def f():
        """ short doc string """
        return 1

    g = epyccel(f, language=language)
    assert(f.__doc__ == g.__doc__)

def test_multiline_line_docstring(language):
    def f():
        """
        Big beautiful doc string

        Parameters
        ----------

        Results
        -------
        1 : int
            no description
        """
        return 1

    g = epyccel(f, language=language)

    python_doc, pyccel_doc = pad_docstrings(f.__doc__, g.__doc__)

    assert python_doc == pyccel_doc


def test_class_docstring(language):
    class A:
        """
        Empty class
        """
        def __init__(self : 'A'):
            pass

    B = epyccel(A, language=language)

    python_doc, pyccel_doc = pad_docstrings(A.__doc__, B.__doc__)
    assert python_doc == pyccel_doc
