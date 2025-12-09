# Testing

While developing it is important to test regularly to start the PR with the best possible code. While some tests are difficult to run locally (e.g. testing across different operating systems) there are several tests that can be run manually before opening a PR.

## Functionality tests

The functionality tests ensure that the code performs as expected. The CI runs lots of these tests across a variety of compilers and operating systems (OS) however you should be able to run locally with your OS and favourite compiler (often GNU). The functionality tests are found in the `tests/` folder and are designed for use with [pytest](https://docs.pytest.org/en/stable/).

Useful flags include:

- `-x` : stop on failure.
- `-v` : display the name of the tests being run.
- `-s` : don't capture output (ensures prints are sent to screen).
- `-k <PATTERN>` : run tests with `<PATTERN>` in the name.
- `-m fortran` : run Fortran tests.
- `-m c` : run C tests.
- `-m python` : run Python tests.
- `-n <NTHREADS>` : run test in parallel with `<NTHREADS>` threads.
- `-m 'not xdist_incompatible'` : run all tests except those which fail when run in parallel.
- `--lf` : run only the tests which failed the last time they were run.
- `--ff` : run the tests which failed the last time they were run before running other tests.

Before opening a PR you should always check that the tests that you have added are working correctly. However if possible it is also good to run the full test suite once as changes to the code can sometimes have an unexpected knock-on effect. The command `pyccel test` runs all tests that appear in the CI.

## Static analysis

The static analysis tests check that the code follows the expected coding conventions. There are multiple tests some of which can be run manually.

### Python Linting

We use [Pylint](https://docs.pylint.org/run.html) to ensure that the code follows good practice for Python. Pylint can be installed using pip. If it is run from the root directory it should automatically find the configuration file `.pylintrc` which specifies the rules that we enforce in Pyccel.

Pylint errors can be ignored locally or globally. In the `tests/` folder we do not enforce documentation of test code so feel free to globally disable Pylint commands related to this by adding the following line to the top of any new files:

```python
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
```

Other than this one exception ignoring errors globally is discouraged. Please fix the error or add a local disable on the problematic line if this is not possible for some reason.

### Pyccel Linting

We have a small number of additional rules that we enforce in Pyccel but which do not appear in Pylint. These tests can be run locally with:

```bash
python3 ci_tools/check_pyccel_conventions.py
```

### Spelling

We use 2 tools to check the spelling. The first is designed to check spelling exactly and is used to check documentation. The second test is looser and is designed to spellcheck code.

If the necessary dictionaries are installed, the first test can be run locally with:

```bash
pip3 install pyspelling
python3 -m pyspelling
```

The necessary dictionaries can be installed using the following command:

```bash
sudo apt install aspell aspell-en
```

When run from the root folder it should automatically detect the configuration file `.pyspelling.yml`.

This test can raise false positives. If you are sure that this is a false positive then you can fix the error by adding the word to the custom dictionary `.dict_custom.txt`. Before doing this please think about the following considerations:

- False positives are often raised for external libraries such as CuPy, SymPy, BLAS. When adding such libraries to the dictionary please ensure that the capitalisation matches that used in their documentation.
- Pyccel uses UK English, if a common word is flagged this may simply be due to a US spelling.
- Code often leads to spelling errors. Code in code tags e.g. `avarname = 2` is not spellchecked. Please ensure that code is correctly quoted.
- Please use a search engine to double check that your word exists before adding it to the dictionary. Often it is better to rephrase sentences using obscure words that are not recognised by the spell checker unless such words are common in the IT community.

The second test uses the [typos](https://github.com/crate-ci/typos) tool. Instructions on how to download and run this tool can be found on its GitHub page.
This test tries to detect typos in code so it handles `snake_case`, `CamelCase`, etc. This can be error-prone, but the tool is designed to reduce false positives. This does however mean that some typos may be missed.

The configuration file is found in `.typos.toml`.

### Markdown Linting

Markdown is checked using [`markdownlint-cli2`](https://github.com/DavidAnson/markdownlint-cli2). Instructions on how to download and run this tool can be found on its GitHub page.

### Documentation

Pyccel uses NumPy formatting for the documentation. The [style guide](https://numpydoc.readthedocs.io/en/latest/format.html) describes the requirements for this format.

Commands exist to check the presence of documentation and the format of this documentation.
Currently documentation is incomplete, so tests only check documentation for new code.

The following command:

```bash
docstr-coverage --config=.docstr.yaml pyccel ci_tools
```

Checks the docstring coverage for the project.

Individual docstrings can be checked using:

```bash
python3 -m numpydoc validate <FUNC_TO_CHECK>
```

E.g:

```bash
python3 -m numpydoc validate pyccel.ast.basic.PyccelAstNode.get_user_nodes
```

The CI automatically runs this command for all functions that have been modified.

Finally the documentation website is generated to ensure that no warnings are raised.

This can be done locally with:

```bash
pip3 install -r docs/requirements.txt
make -C docs html
```

The documentation will then be found in `docs/build/html`.
