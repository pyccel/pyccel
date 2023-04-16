# pylint: disable=missing-function-docstring, missing-module-docstring
from pytest_teardown_tools import run_epyccel, clean_test

def test_1_line_docstring(language):
    def f():
        """ short doc string """
        return 1

    g = run_epyccel(f, language=language)
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

    g = run_epyccel(f, language=language)

    # Remove empty lines as ast does not preserve them
    python_doc = [p for p in f.__doc__.split('\n') if p.strip()]
    pyccel_doc = [p for p in g.__doc__.split('\n') if p.strip()]

    # Pad the smaller doc string to ensure a match
    extra_spaces = len(python_doc[0]) - len(pyccel_doc[0])
    if extra_spaces>0:
        pyccel_doc = [' '*extra_spaces+p for p in pyccel_doc]
    if extra_spaces<0:
        extra_spaces = -extra_spaces
        python_doc = [' '*extra_spaces+p for p in python_doc]

    python_doc = '\n'.join(python_doc)
    pyccel_doc = '\n'.join(pyccel_doc)
    assert(python_doc == pyccel_doc)




##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
