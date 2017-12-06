# -*- coding: utf-8 -*-
"""
    Quickly setup a Pyccel source to work with pyccel.
"""

import sys
import os
import argparse

from os import path

class MyParser(argparse.ArgumentParser):
    """
    Custom argument parser for printing help message in case of an error.
    See http://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def mkdir_p(dir):
    # type: (unicode) -> None
    if path.isdir(dir):
        return
    os.makedirs(dir)

def generate(d, overwrite=True, silent=False):
    """Generates the project from a dictionary."""
    # escape backslashes and single quotes in strings that are put into
    # a Python string literal
    for key in ('project', 'project_doc', 'project_doc_texescaped',
                'author', 'author_texescaped', 'copyright',
                'version', 'release', 'master'):
        d[key + '_str'] = d[key].replace('\\', '\\\\').replace("'", "\\'")

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
    mkdir_p(path.join(srcdir, d['dot'] + 'templates'))
    mkdir_p(path.join(srcdir, d['dot'] + 'static'))

def main():
    """Creates a new pyccel project."""
    parser = MyParser(description='pyccel-quickstart command line')

    # ...
    parser.add_argument('--sep', action='store_true', \
                        help='if specified, separate source and build dirs.')
    parser.add_argument('--quiet', action='store_true', \
                        help='quiet mode.')
    # ...

    # ... project basic options
    parser.add_argument('--project', type=str, \
                        help='project name')
    parser.add_argument('--author', type=str, \
                        help='author name')
    parser.add_argument('-v', type=str, \
                        help='version of project')
    parser.add_argument('-r', type=str, \
                        help='release of project')
    # ...

    # ... project advanced options
    parser.add_argument('-l', type=str, \
                        help='language of project')
#    parser.add_argument('--language', type=str, \
#                        help='Target language')
    parser.add_argument('--compiler', type=str, \
                        help='Used compiler')

    parser.add_argument('--include', type=str, \
                        help='path to include directory.')
    parser.add_argument('--libdir', type=str, \
                        help='path to lib directory.')
    parser.add_argument('--libs', type=str, \
                        help='list of libraries to link with.')
    # ...

    # ... Extension options
    parser.add_argument('--ext-linalg', action='store_true', \
                        help='enable linear algebra extension.')

    parser.add_argument('--extensions', type=str, \
                        help='enable extensions.')
    # ...

    # ...
    args = parser.parse_args()
    # ...
    print args

