# -*- coding: utf-8 -*-
"""
    Quickly setup a Pyccel source to work with pyccel.
"""

# TODO this file has not been refactored yet


import sys
import os
import argparse

from os import path

from pyccel import __version__ as __display_version__
from pyccel.codegen.utilities_old import build_file
from pyccel.codegen.utilities_old import build_cmakelists
from pyccel.codegen.utilities_old import build_cmakelists_dir
from pyccel.codegen.utilities_old import generate_project_main


EXTENSIONS = {
    'math': True,
    'blas': False
}

DEFAULT_VALUE = {
    'author': '__AUTHOR__',
    'sep': True,
    'language': 'fortran',
    'suffix': '.f90',
    'master': 'main',
}


def mkdir_p(dir):
    # type: (unicode) -> None
    if path.isdir(dir):
        return
    os.makedirs(dir)


def get_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        usage='usage: %(prog)s [OPTIONS] SOURCEDIR [FILENAMES...]',
        epilog='For more information, visit <http://http://pyccel.readthedocs.io/>.',
        description="""
Generate low-level code from python source files.
pyccel-build generates low-level code from the files in SOURCEDIR and places it
in OUTPUT_DIR. It looks for 'conf.py' in SOURCEDIR for the configuration
settings.  The 'pyccel-quickstart' tool may be used to generate template files,
including 'conf.py'
pyccel-build can create low-level code in different languages. A language is
selected by specifying the builder name on the command line; it defaults to
FORTRAN.
By default, everything that is outdated is built. Output only for selected
files can be built by specifying individual filenames.
""")

    parser.add_argument('--version', action='version', dest='show_version',
                        version='%%(prog)s %s' % __display_version__)

    parser.add_argument('sourcedir',
                        help='path to pyccel source files')
    parser.add_argument('filenames', nargs='*',
                        help='a list of specific files to rebuild. Ignored '
                        'if -a is specified')

    group = parser.add_argument_group('general options')
    group.add_argument('--output-dir', type=str, \
                        help='Output directory.')
    group.add_argument('--convert-only', action='store_true',
                       help='Converts pyccel files only without build')
    group.add_argument('-b', metavar='BUILDER', dest='builder',
                       default='fortran',
                       help='builder to use (default: fortran)')
    group.add_argument('-a', action='store_true', dest='force_all',
                       help='write all files (default: only write new and '
                       'changed files)')
    group.add_argument('-E', action='store_true', dest='freshenv',
                       help='don\'t use a saved environment, always read '
                       'all files')
    group.add_argument('-j', metavar='N', default=1, type=int, dest='jobs',
                       help='build in parallel with N processes where '
                       'possible')

    group = parser.add_argument_group('build configuration options')
    group.add_argument('-c', metavar='PATH', dest='confdir',
                       help='path where configuration file (conf.py) is '
                       'located (default: same as SOURCEDIR)')
    group.add_argument('-D', metavar='setting=value', action='append',
                       dest='define', default=[],
                       help='override a setting in configuration file')

    group = parser.add_argument_group('console output options')
    group.add_argument('-v', action='count', dest='verbosity', default=0,
                       help='increase verbosity (can be repeated)')
    group.add_argument('-q', action='store_true', dest='quiet',
                       help='no output on stdout, just warnings on stderr')
    group.add_argument('-Q', action='store_true', dest='really_quiet',
                       help='no output at all, not even warnings')
    group.add_argument('-W', action='store_true', dest='warningiserror',
                       help='turn warnings into errors')

    return parser


# TODO default debug should be True for the moment
def build(d, silent=False, force=True,
          dep_libs=[], dep_extensions=['math'],
          clean=True, debug=True):
    """Generates the project from a dictionary."""
    if not debug:
        sys.tracebacklimit = 0

    conf_filename = os.path.join(os.getcwd(), 'conf.py')
    if not os.path.exists(conf_filename):
        raise ValueError('Could not find conf.py file.'
                        ' Make sure you run pyccel-build from the right directory.')

    language   = d['language']
    sourcedir  = d['sourcedir']
    output_dir = d['output_dir']

    mkdir_p(output_dir)

    py_file     = lambda f: (f.split('.')[-1] == 'py')
    ignore_file = lambda f: (os.path.basename(f) in ['__init__.py'])

    files = [f for f in os.listdir(sourcedir) if py_file(f) and not ignore_file(f)]

    programs = []
    for f_name in files:
        if not silent:
            print ('> converting {0}/{1}'.format(sourcedir, f_name))

        filename = os.path.join(sourcedir, f_name)
        info = build_file(filename, language=language, compiler=None, output_dir=output_dir)
        if not info['is_module']:
            programs.append(f_name.split('.')[0])

    # remove .pyccel temporary files
    if clean:
        os.system('rm {0}/*.pyccel'.format(output_dir))

    # ...
    is_program = lambda f: (f.split('.')[0] in programs)
    valid_file = lambda f: (f.split('.')[-1] in ['f90'])

    files    = [f for f in os.listdir(output_dir) if valid_file(f) if not(is_program(f))]
    programs = [f for f in os.listdir(output_dir) if valid_file(f) if is_program(f)]
#    print('>>>> files    = {0}'.format(files))
#    print('>>>> programs = {0}'.format(programs))

    libname = os.path.basename(sourcedir)

    dep_libs += ['pyccelext_{0}'.format(i) for i in dep_extensions]
    build_cmakelists(output_dir, libname, files,
                     force=force, dep_libs=dep_libs,
                     programs=programs)

    build_cmakelists_dir('src')
    # ...

    # ...
    if not('convert_only' in d):
        from pyccel.codegen.cmake import CMake
        cmake = CMake('.')
        cmake.make()
    # ...

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
    settings['extensions'] = [k[4:] for k,v in d.items() if v and (k[:4] == 'ext_')]
    # ...

    # ... default value is ./src/SRCDIR
    if not 'output_dir' in settings:
        srcdir = settings['sourcedir']

        settings['output_dir'] = 'src/{0}'.format(srcdir)
    # ...

    # ...
    build(settings)
    # ...
