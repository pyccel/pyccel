"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
from itertools import chain
import pathlib
import sys

from pygments.lexers.fortran import FortranLexer

pyccel_dir = pathlib.Path(__file__).parent.parent.parent

sys.path.append(str(pyccel_dir.resolve()))

def setup(app):
    """
    Override the default 'fortran' lexer with one from Pygments which is
    better at parsing the code.
    """
    from sphinx.highlighting import lexers
    lexers['fortran'] = FortranLexer()

# -- Project information -----------------------------------------------------

project = 'pyccel'
author = '*'

# The full version, including alpha/beta/rc tags
release = '*'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        "sphinx.ext.viewcode",
        "sphinx.ext.doctest",
        "sphinx.ext.napoleon", #NumPy style docstrings
        "sphinx_github_style",
        "myst_parser",
        ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

suppress_warnings = [
        "autodoc.import_object",
        ]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'renku'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for myst_parser -------------------------------------------------
myst_heading_anchors = 3

# -- Options for sphinx_github_style -----------------------------------------
linkcode_url = 'https://github.com/pyccel/pyccel'

# -- Link fixing -------------------------------------------------------------
base_dir = pyccel_dir / 'docs' / 'source'
api_dir = base_dir / 'api'
python_path_substitutions = {}
for module in api_dir.iterdir():
    path_name = module.stem.replace('.','/')
    new_link = f'../api/{module.name}'
    if (pyccel_dir / (path_name + '.py')).exists():
        python_path_substitutions[f'../{path_name}.py'] = new_link
    else:
        python_path_substitutions[f'../{path_name})'] = new_link
        python_path_substitutions[f'../{path_name}/)'] = new_link

for doc_files in chain((base_dir / 'docs').iterdir(), (base_dir / 'developer_docs').iterdir()):
    with open(doc_files, 'r', encoding="utf-8") as f:
        contents = f.read()
    for file_path, api_mod_path in python_path_substitutions.items():
        contents = contents.replace(file_path, api_mod_path)
    contents = contents.replace('../', 'https://github.com/pyccel/pyccel/tree/devel/')
    with open(doc_files, 'w', encoding="utf-8") as f:
        f.write(contents)
