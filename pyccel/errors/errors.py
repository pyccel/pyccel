#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
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

from pyccel.ast.basic import PyccelAstNode
from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.stage import PyccelStage

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

pyccel_stage = PyccelStage()

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
    """
    Representation of a single error message.

    A class which holds all of the information necessary to describe
    an error message raised by Pyccel.

    Parameters
    ----------
    stage : str
        The Pyccel stage when the error occurred.

    filename : str
        The file where the error was detected.

    message : str
        The message to be displayed to the user.

    line : int, optional
        The line where the error was detected.

    column : int, optional
        The column in the line of code where the error was detected.

    severity : str, optional
        The severity of the error. This is one of : [warning/error/fatal].

    symbol : str, optional
        A string representation of the PyccelAstNode object which caused
        the error to need to be raised.
        This object is printed in the error message.

    traceback : str, optional
        The traceback describing the execution of the code when the error
        was raised.
    """

    def __init__(self, *, stage, filename,
                 message,
                 line=None,
                 column=None,
                 severity=None,
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
        self.blocker = (ErrorsMode().value == 'developer' and severity != 'warning' and 'raise ' not in traceback) \
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
                info['location'] = ' [{line},{column}]'.format(line=self.line, column=self.column)
            else:
                info['location'] = ' [{line}]'.format(line=self.line)

        if self.symbol:
            info['symbol'] = f' ({self.symbol})'

        return pattern.format(**info)


class ErrorsMode(metaclass = Singleton):
    """
    The mode for the error output.

    The mode for the error output. This is either 'developer' or 'user'.
    In developer mode the errors are more verbose and include a traceback
    this helps developers debug errors.
    """
    def __init__(self):
        self._mode = 'user'

    @property
    def value(self):
        return self._mode

    def set_mode(self, mode):
        """
        Set the error mode.

        Set the error mode to either 'developer' or 'user'.

        Parameters
        ----------
        mode : str
            The new error mode.
        """
        assert mode in ['user', 'developer']
        self._mode = mode


class Errors(metaclass = Singleton):
    """
    Container for compile errors.

    A singleton class which contains all functions necessary to
    raise neat user-friendly errors in Pyccel.
    """
    _stage_names = {'syntactic': 'parsing (syntax)',
                    'semantic': 'annotation (semantic)',
                    'printing': 'code generation',
                    'cwrapper': 'code generation (wrapping)',
                    'compilation': 'compilation',
                    'buildgen': 'build system generation'}

    def __init__(self):
        self.error_info_map = None
        self._target = None
        self._mode = ErrorsMode()

        self.initialize()

    @property
    def target(self):
        return self._target

    @property
    def mode(self):
        return self._mode.value

    def initialize(self):
        """
        Initialise the Errors singleton.

        Initialise the Errors singleton. This function is necessary so
        the singleton can be reinitialised using the `reset` function.
        """
        self.error_info_map = OrderedDict()

        self._target = None

    def reset(self):
        """
        Reset the Errors singleton.

        Reset the Errors singleton. This removes any information about
        previously generated errors or warnings. This method should be
        called before starting a new translation.
        """
        self.initialize()

    def set_target(self, target):
        """
        Set the current translation target.

        Set the current translation target which describes the location
        from which the error is being raised.

        Parameters
        ----------
        target : str
            The name of the file being translated.
        """
        self._target = target

    def report(self,
               message,
               line = None,
               column = None,
               bounding_box = None,
               severity = 'error',
               symbol = None,
               filename = None,
               traceback = None):
        """
        Report an error.

        Report message at the given line using the current error context
        stage: 'syntactic', 'semantic' 'codegen', or 'cwrapper'.

        Parameters
        ----------
        message : str
            The message to be displayed to the user.

        line : int, optional
            The line at which the error can be found.
            Default: If a symbol is provided with a known line number then this line number is used.

        column : int, optional
            The column at which the error can be found.
            Default: If a symbol is provided with a known column then this column is used.

        bounding_box : tuple, optional
            An optional tuple containing the line and column.

        severity : str, default='error'
            Indicates the seriousness of the error. Should be one of: 'warning', 'error', 'fatal'.
            Default: 'error'.

        symbol : pyccel.ast.PyccelAstNode, optional
            The PyccelAstNode object which caused the error to need to be raised.
            This object is printed in the error message.

        filename : str, optional
            The file which was being treated when the error was found.

        traceback : types.TracebackType
            The traceback that was raised when the error appeared.
        """
        # filter internal errors
        if (self.mode == 'user') and (severity == 'internal'):
            return

        if filename is None:
            filename = self.target

        # TODO improve. it is assumed here that tl and br have the same line
        if bounding_box:
            line   = bounding_box[0]
            column = bounding_box[1]

        ast_node = None

        if symbol is not None:
            if isinstance(symbol, ast.AST):
                ast_node = symbol
                symbol = ast.unparse(ast_node)
            elif isinstance(symbol, PyccelAstNode):
                ast_node = symbol.python_ast

            if self.mode == 'developer':
                symbol = repr(symbol)
            else:
                symbol = str(symbol)

        if ast_node:
            if line is None:
                line   = getattr(ast_node, 'lineno', None)
            if column is None:
                column = getattr(ast_node, 'col_offset', None)

        if self.mode == 'developer':
            if traceback:
                traceback = ''.join(tb.format_tb(traceback, limit=-5))
            else:
                traceback = ''.join(tb.format_stack(limit=5))
        else:
            traceback = None

        info = ErrorInfo(stage=pyccel_stage.current_stage,
                         filename=filename,
                         message=message,
                         line=line,
                         column=column,
                         severity=severity,
                         symbol=symbol,
                         traceback=traceback)

        self.add_error_info(info)

        if info.blocker:
            if pyccel_stage == 'syntactic':
                raise PyccelSyntaxError(message)
            elif pyccel_stage == 'semantic':
                raise PyccelSemanticError(message)
            elif pyccel_stage == 'codegen':
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

    def __str__(self):
        text = ''
        errors = [err for err_list in self.error_info_map.values() for err in err_list if err.severity != 'warning']
        if errors:
            stage = next(e.stage for e in errors if e.severity != 'warning')
            if stage is not None:
                text += f'\nERROR at {self._stage_names[stage]} stage\n'

        print_path = (len(self.error_info_map.keys()) > 1)
        if self.error_info_map:
            text += f'{PYCCEL}:\n'

        for path in self.error_info_map.keys():
            errors = self.error_info_map[path]
            if print_path: text += ' filename :: {path}\n'.format(path=path)
            for err in errors:
                text += ' ' + str(err) + '\n'

        return text

