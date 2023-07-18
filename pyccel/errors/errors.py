#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains classes and methods that manipilate the various errors and warnings
that could be shown by pyccel.
"""
import ast
import sys
import traceback as tb

from collections import OrderedDict
from os.path import basename

from pyccel.ast.basic import Basic
from pyccel.utilities.metaclasses import Singleton

# ...
#ERROR = 'error'
#INTERNAL = 'internal'
#WARNING = 'warning'
#FATAL = 'fatal'
#
#PYCCEL = 'pyccel'
#
#def make_symbol(s):
#    return str(s)

try:
    from termcolor import colored
    ERROR = colored('error', 'red', attrs=['blink', 'bold'])
    INTERNAL = colored('internal', attrs=['blink', 'bold'])
    WARNING = colored('warning', 'green', attrs=['blink'])
    FATAL = colored('fatal', 'red', attrs=['blink', 'bold'])

    PYCCEL = colored('pyccel', attrs=['bold'])

    def make_symbol(s):
        return colored(str(s), attrs=['bold'])
except ImportError:
    ERROR = 'error'
    INTERNAL = 'internal'
    WARNING = 'warning'
    FATAL = 'fatal'

    PYCCEL = 'pyccel'

    def make_symbol(s):
        return str(s)
# ...

_severity_registry = {'error': ERROR,
                      'internal': INTERNAL,
                      'fatal': FATAL,
                      'warning': WARNING}


class PyccelError(Exception):
    def __init__(self, message, errors=''):

        # Call the base class constructor with the parameters it needs
        super(PyccelError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

class PyccelSyntaxError(PyccelError):
    pass

class PyccelSemanticError(PyccelError):
    pass

class PyccelCodegenError(PyccelError):
    pass


class ErrorInfo:
    """Representation of a single error message."""

    def __init__(self, *, stage, filename,
                 line=None,
                 column=None,
                 severity=None,
                 message='',
                 symbol=None,
                 traceback=None):
        # The parser stage
        self.stage = stage
        # The source file that was the source of this error.
        self.filename = basename(filename) if filename is not None else ''
        # The line number related to this error within file.
        self.line = line
        # The column number related to this error with file.
        if isinstance(column, (tuple, list)):
            column = '-'.join(str(i) for i in column)
        self.column = column
        # Either 'error', 'fatal', or 'warning'.
        self.severity = severity
        # The error message.
        self.message = message
        # Symbol associated to the message
        self.symbol = symbol
        # If True, we should halt build after the file that generated this error.
        self.blocker = (ErrorsMode().value == 'developer' and severity != 'warning') \
                or (severity == 'fatal')
        # The traceback at the moment that the error was raised
        self.traceback = traceback

    def __str__(self):

        pattern = '{traceback}|{severity} [{stage}]: {filename}{location}| {message}{symbol}'
        info = {
            'stage'   : self.stage,
            'severity': _severity_registry[self.severity],
            'filename': self.filename,
            'location': '',
            'message' : self.message,
            'symbol'  : '',
            'traceback': self.traceback or ''
        }

        if self.line:
            if self.column:
                info['location'] = f' [{self.line},{self.column}]'
            else:
                info['location'] = f' [{self.line}]'

        if self.symbol:
            if self.traceback:
                info['symbol'] = f' ({repr(self.symbol)})'
            else:
                info['symbol'] = f' ({self.symbol})'

        return pattern.format(**info)


class ErrorsMode(metaclass = Singleton):
    """Developper or User mode.
    pyccel command line will set it.
    """
    def __init__(self):
        self._mode = 'user'

    @property
    def value(self):
        return self._mode

    def set_mode(self, mode):
        assert(mode in ['user', 'developer'])
        self._mode = mode


class Errors(metaclass = Singleton):
    """Container for compile errors.
    """

    def __init__(self):
        self.error_info_map = None
        self._target = None
        self._parser_stage = None
        self._mode = ErrorsMode()

        self.initialize()

    @property
    def parser_stage(self):
        return self._parser_stage

    @property
    def target(self):
        return self._target

    @property
    def mode(self):
        return self._mode.value

    def initialize(self):
        self.error_info_map = OrderedDict()

        self._target = {}
        self._target['file'] = None
        self._target['module'] = None
        self._target['function'] = None
        self._target['class'] = None

    def reset(self):
        self.initialize()

    def set_parser_stage(self, stage):
        assert(stage in ['syntax', 'semantic', 'codegen'])
        self._parser_stage = stage

    def set_target(self, target, kind):
        assert(kind in ['file', 'module', 'function', 'class'])
        self._target[kind] = target

    def unset_target(self, kind):
        assert(kind in ['file', 'module', 'function', 'class'])
        self._target[kind] = None

    def reset_target(self):
        """."""
        self._target = {}
        self._target['file'] = None
        self._target['module'] = None
        self._target['function'] = None
        self._target['class'] = None

    def report(self,
               message,
               line = None,
               column = None,
               bounding_box = None,
               severity = 'error',
               symbol = None,
               filename = None,
               verbose = False):
        """
        Report message at the given line using the current error context.
        stage: 'syntax', 'semantic' or 'codegen'

        Parameters
        ----------
        message : str
                  The message to be displayed to the user
        line    : int
                  The line at which the error can be found
                  Default: If a symbol is provided with a known line number
                  then this line number is used
        column  : int
                  The column at which the error can be found
                  Default: If a symbol is provided with a known column
                  then this column is used
        bounding_box : tuple
                  An optional tuple containing the line and column
        severity : str
                  Indicates the seriousness of the error. Should be one of:
                  'warning', 'error', 'fatal'
                  Default: 'error'
        symbol   : pyccel.ast.Basic
                  The Basic object which caused the error to need to be raised.
                  This object is printed in the error message
        filename : str
                  The file which was being treated when the error was found
        verbose  : bool
                  Flag to add verbosity
                  Default: False
        """
        # filter internal errors
        if (self.mode == 'user') and (severity == 'internal'):
            return

        if filename is None:
            filename = self.target['file']

        # TODO improve. it is assumed here that tl and br have the same line
        if bounding_box:
            line   = bounding_box[0]
            column = bounding_box[1]

        fst = None

        if symbol is not None:
            if isinstance(symbol, ast.AST):
                fst = symbol
                if sys.version_info < (3, 9):
                    symbol = ast.dump(fst)
                else:
                    symbol = ast.unparse(fst) # pylint: disable=no-member
            elif isinstance(symbol, Basic):
                fst = symbol.fst

        if fst:
            line   = getattr(fst, 'lineno', None)
            column = getattr(fst, 'col_offset', None)

        traceback = None
        if self.mode == 'developer':
            traceback = ''.join(tb.format_stack(limit=5))

        info = ErrorInfo(stage=self._parser_stage,
                         filename=filename,
                         line=line,
                         column=column,
                         severity=severity,
                         message=message,
                         symbol=symbol,
                         traceback=traceback)

        if verbose: print(info)

        self.add_error_info(info)

        if info.blocker:
            if self._parser_stage == 'syntax':
                raise PyccelSyntaxError(message)
            elif self._parser_stage == 'semantic':
                raise PyccelSemanticError(message)
            elif self._parser_stage == 'codegen':
                raise PyccelCodegenError(message)
            else:
                raise PyccelError(message)

    def _add_error_info(self, file, info):
        if file not in self.error_info_map:
            self.error_info_map[file] = []
        self.error_info_map[file].append(info)

    def add_error_info(self, info):
        self._add_error_info(info.filename, info)

    def num_messages(self):
        """Return the number of generated messages."""
        return sum(len(x) for x in self.error_info_map.values())

    def has_warnings(self):
        """Are there any errors that are warnings?"""
        return any(err for errs in self.error_info_map.values() for err in errs if err.severity == 'warning')

    def has_errors(self):
        """Are there any generated errors?"""
        return any(err for errs in self.error_info_map.values() for err in errs if err.severity != 'warning')

    def has_blockers(self):
        """Are there any errors that are blockers?"""
        return any(err for errs in self.error_info_map.values() for err in errs if err.blocker)

    def blocker_filename(self):
        """Return the file with a blocking error, or None if not possible."""
        for errs in self.error_info_map.values():
            for err in errs:
                if err.blocker:
                    return err.filename
        return None

    def format_messages(self, error_info):
        """Return a string list that represents the error messages.
        Use a form suitable for displaying to the user.
        """
        pass

    def check(self):
        """."""
        if self.num_messages() > 0:
            print(self.__str__())

    def __str__(self):
        print_path = (len(self.error_info_map.keys()) > 1)
        text = f'{PYCCEL}:\n'

        for path in self.error_info_map.keys():
            errors = self.error_info_map[path]
            if print_path: text += f' filename :: {path}\n'
            for err in errors:
                text += ' ' + str(err) + '\n'

        return text

