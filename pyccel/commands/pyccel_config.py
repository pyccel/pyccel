#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing scripts to manage compilation information.
"""
import json
import pathlib
import shutil
import sys

from .argparse_helpers import add_help_flag, path_with_suffix, add_compiler_selection

__all__ = ('pyccel_config',
           'setup_pyccel_config_parser',
           'PYCCEL_CONFIG_DESCR')

PYCCEL_CONFIG_DESCR = 'Compilation configuration management.'

def setup_pyccel_config_parser(parser):
    """
    Add the `pyccel config` arguments to the parser.

    Add the `pyccel config` arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    subparsers = parser.add_subparsers(required=True, title='Subcommands', metavar='COMMAND')
    export_parser = subparsers.add_parser('export', add_help=False, help="Export a compiler configuration to a json file.")
    export_parser.add_argument('filename', metavar='FILE', type=path_with_suffix(('.json',), must_exist = False),
                        help='The file that the parser information should be exported to.')
    export_parser.set_defaults(config_func=pyccel_config)

    check_parser = subparsers.add_parser('check', add_help=False, help="Check that a compiler configuration is valid.")
    check_parser.add_argument('filename', metavar='FILE', type=path_with_suffix(('.json',)),
                        help='The file containing the compiler configuration.')
    check_parser.set_defaults(config_func=pyccel_config_check)

    register_parser = subparsers.add_parser('register', add_help=False, help="Register a commonly used compiler configuration.")
    register_parser.add_argument('compiler_family', metavar='FAMILY', type=str,
                                 help='The name that will be used to identify the compiler.')
    register_parser.add_argument('filename', metavar='FILE', type=path_with_suffix(('.json',)),
                        help='The file containing the compiler configuration.')
    register_parser.add_argument('-v', '--verbose', action='count', default = 0,
                        help='Increase output verbosity (use -v, -vv, -vvv for more detailed output).')
    register_parser.add_argument('--conda-warnings', choices=('off', 'basic', 'verbose'), default='basic',
                        help='Specify the level of Conda warnings to display (default: basic).')
    register_parser.set_defaults(config_func=pyccel_config_register)

    remove_parser = subparsers.add_parser('remove', add_help=False, help="Remove a register compiler configuration.")
    remove_parser.add_argument('compiler_family', metavar='FAMILY', type=str,
                           help='The name that identifies the compiler.')
    remove_parser.set_defaults(config_func=pyccel_remove_config)

    # ... Compiler options
    add_compiler_selection(export_parser)
    add_help_flag(export_parser.add_argument_group('Options'))
    add_help_flag(check_parser.add_argument_group('Options'))
    add_help_flag(register_parser.add_argument_group('Options'))

    add_help_flag(parser.add_argument_group('Options'))

def pyccel_config_dispatch(config_func, **kwargs):
    config_func(**kwargs)

def pyccel_config(filename, **kwargs):
    """
    Call the `pyccel config` pipeline.

    Import and call the `pyccel config` pipeline.

    Parameters
    ----------
    filename : Path
        Name of the JSON file where an exported configuration is printed.
    **kwargs : dict
        See execute_pyccel.
    """
    from pyccel.codegen.pipeline  import execute_pyccel
    execute_pyccel('', compiler_export_file = filename, **kwargs)

def pyccel_config_check(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        try:
            config_contents = json.load(fp)
        except json.JSONDecodeError:
            print("File is not in json format", file=sys.stderr)
            sys.exit(1)

    from pyccel.compilers.default_compilers import available_compilers

    example_compiler = available_compilers['GNU']

    languages = example_compiler.keys()

    if all(k not in languages for k in config_contents):
        print("First level key should describe languages.", file=sys.stderr)
        sys.exit(1)

    exitcode = 0
    for k in config_contents:
        if k not in languages:
            print("Unrecognised language :", k)
            exitcode = 1

    if exitcode:
        sys.exit(1)

    for lang, lang_config in config_contents.items():
        example_config = example_compiler[lang]
        possible_keys = {k for k,v in example_config.items() if not isinstance(v, dict)}
        possible_accelerators = {k for k,v in example_config.items() if isinstance(v, dict)}
        found_keys = set()
        found_accelerators = set()

        for k,v in lang_config.items():
            if k in possible_keys:
                found_keys.add(k)
            elif k in possible_accelerators:
                found_accelerators.add(k)
            else:
                print(f"Warning: Key {k} in language {lang} is unrecognised")

        for k in possible_keys.difference(found_keys):
            print(f"Warning: Key {k} not provided for language {lang}. It will default to an empty value")

        for k in found_keys:
            if not isinstance(lang_config[k], type(example_config[k])):
                print("Error: Key {k} in language {lang} is associated with a value of the wrong type.")
                print("Received:", lang_config[k])
                if isinstance(example_config[k], str):
                    print("Expected: str")
                else:
                    print("Expected: tuple[str]")
                exitcode = 1

    if exitcode:
        sys.exit(exitcode)

def pyccel_config_register(compiler_family, filename, verbose, conda_warnings):
    pyccel_config_check(filename)

    from pyccel.codegen.compiling.library_config import recognised_libs
    from pyccel.codegen.compiling.compilers import Compiler, get_condaless_search_path
    from pyccel import __version__ as pyccel_version

    installed_libs = {}

    compiler = Compiler(str(filename))

    config_dirpath = pathlib.Path.home() / '.pyccel' / compiler_family

    with open(filename, 'r', encoding='utf-8') as fp:
        config_contents = json.load(fp)

    config_contents['pyccel_version'] = pyccel_version

    try:
        config_dirpath.mkdir(parents = True)
    except FileExistsError:
        print("A compiler with the chosen compiler family name is already registered.")
        sys.exit(1)

    with open(config_dirpath / 'config.json', 'w', encoding='utf-8') as fp:
        json.dump(config_contents, fp, indent=2)

    Compiler.acceptable_bin_paths = get_condaless_search_path(conda_warnings)

    recognised_libs['stc'].install_to(config_dirpath, installed_libs, verbose, compiler)

    shutil.rmtree(config_dirpath / 'STC' / f'build-{filename.stem}')

def pyccel_remove_config(compiler_family):
    config_dirpath = pathlib.Path.home() / '.pyccel' / compiler_family
    if config_dirpath.exists():
        shutil.rmtree(str(config_dirpath))
    else:
        print(f"Configuration not found : {compiler_family}")
        sys.exit(1)
