# -*- coding: utf-8 -*-
"""
    Quickly setup a Pyccel source to work with pyccel.
"""

import sys
import os
import argparse

from os import path

EXTENSIONS = ('blas', 'linalg')

DEFAULT_VALUE = {
    'author': '__AUTHOR__',
    'sep': False,
    'dot': '_',
    'language': 'fortran',
    'suffix': '.f90',
    'master': 'main',
    'epub': False,
    'ext_autodoc': False,
    'ext_doctest': False,
    'ext_todo': False,
    'makefile': True,
    'batchfile': True,
}


def mkdir_p(dir):
    # type: (unicode) -> None
    if path.isdir(dir):
        return
    os.makedirs(dir)


def get_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        usage='%(prog)s [OPTIONS] <PROJECT_DIR>',
        epilog="For more information, visit <http://pyccel.readthedocs.io/>.",
        description="""
Generate required files for a Sphinx project.

pyccel-quickstart is an interactive tool that asks some questions about your
project and then generates a complete pyccel directory and sample
Makefile to be used with pyccel-build.
""")

    parser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                        default=False,
                        help='quiet mode')
#    parser.add_argument('--version', action='version', dest='show_version',
#                        version='%%(prog)s %s' % __display_version__)

    parser.add_argument('path', metavar='PROJECT_DIR', default='.',
                        help='output path')

    group = parser.add_argument_group('Structure options')
    group.add_argument('--sep', action='store_true',
                       help='if specified, separate source and build dirs')
    group.add_argument('--dot', metavar='DOT',
                       help='replacement for dot in _build etc.')

    group = parser.add_argument_group('Project basic options')
    group.add_argument('-a', '--author', metavar='AUTHOR', dest='author',
                       help='author names')
    group.add_argument('-v', metavar='VERSION', dest='version', default='',
                       help='version of project')
    group.add_argument('-r', '--release', metavar='RELEASE', dest='release',
                       help='release of project')
    group.add_argument('-l', '--language', metavar='LANGUAGE', dest='language',
                       help='target language')
    parser.add_argument('--compiler', type=str, \
                        help='Used compiler')
    group.add_argument('--master', metavar='MASTER',
                       help='master document name')
    parser.add_argument('--include', type=str, \
                        help='path to include directory.')
    parser.add_argument('--libdir', type=str, \
                        help='path to lib directory.')
    parser.add_argument('--libs', type=str, \
                        help='list of libraries to link with.')

    group = parser.add_argument_group('Extension options')
    for ext in EXTENSIONS:
        group.add_argument('--ext-' + ext, action='store_true',
                           dest='ext_' + ext, default=False,
                           help='enable %s extension' % ext)
    group.add_argument('--extensions', metavar='EXTENSIONS', dest='extensions',
                       action='append', help='enable extensions')

#    # TODO(stephenfin): Consider using mutually exclusive groups here
#    group = parser.add_argument_group('Makefile and Batchfile creation')
#    group.add_argument('--makefile', action='store_true', default=False,
#                       help='create makefile')
#    group.add_argument('--no-makefile', action='store_true', default=False,
#                       help='not create makefile')
#    group.add_argument('--batchfile', action='store_true', default=False,
#                       help='create batchfile')
#    group.add_argument('--no-batchfile', action='store_true', default=False,
#                       help='not create batchfile')
#    group.add_argument('-M', '--no-use-make-mode', action='store_false',
#                       dest='make_mode', default=False,
#                       help='not use make-mode for Makefile/make.bat')
#    group.add_argument('-m', '--use-make-mode', action='store_true',
#                       dest='make_mode', default=True,
#                       help='use make-mode for Makefile/make.bat')

    return parser


def generate(d, overwrite=True, silent=False):
    """Generates the project from a dictionary."""
    # escape backslashes and single quotes in strings that are put into
    # a Python string literal
#    for key in ('project',
#                'author', 'copyright',
#                'version', 'release', 'master'):

    if not path.isdir(d['path']):
        mkdir_p(d['path'])

    srcdir = d['sep'] and path.join(d['path'], 'source') or d['path']

    mkdir_p(srcdir)
    if d['sep']:
        builddir = path.join(d['path'], 'build')
        d['exclude_patterns'] = ''
    else:
        builddir = path.join(srcdir, d['dot'] + 'build')
        exclude_patterns = map(repr, [
            d['dot'] + 'build',
            'Thumbs.db', '.DS_Store',
        ])
        d['exclude_patterns'] = ', '.join(exclude_patterns)
    mkdir_p(builddir)
    mkdir_p(path.join(srcdir, 'extensions'))
    mkdir_p(path.join(srcdir, 'external'))
    mkdir_p(path.join(srcdir, 'src'))

def main(argv=sys.argv[1:]):
    """Creates a new pyccel project."""
    # ...
    parser = get_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        return err.code
    # ...

    d = vars(args)
    # delete None or False value
    d = dict((k, v) for k, v in d.items() if not (v is None or v is False))

    # ...
    settings = DEFAULT_VALUE.copy()

    for k,v in d.items():
        settings[k] = v
    # ...

    # ...
    generate(settings)
    # ...
