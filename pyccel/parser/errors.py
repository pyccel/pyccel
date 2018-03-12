from collections import OrderedDict

# Constants that represent simple type checker error message, i.e. messages
# that do not have any parameters.

UNDEFINED_VARIABLE = 'Undefined variable'

NO_RETURN_VALUE_EXPECTED = 'No return value expected'
MISSING_RETURN_STATEMENT = 'Missing return statement'
INVALID_IMPLICIT_RETURN = 'Implicit return in function which does not return'
INCOMPATIBLE_RETURN_VALUE_TYPE = 'Incompatible return value type'
RETURN_VALUE_EXPECTED = 'Return value expected'
NO_RETURN_EXPECTED = 'Return statement in function which does not return'
INCOMPATIBLE_TYPES = 'Incompatible types'
INCOMPATIBLE_TYPES_IN_ASSIGNMENT = 'Incompatible types in assignment'
INCOMPATIBLE_REDEFINITION = 'Incompatible redefinition'

INCOMPATIBLE_TYPES_IN_STR_INTERPOLATION = 'Incompatible types in string interpolation'
MUST_HAVE_NONE_RETURN_TYPE = 'The return type of "{}" must be None'
INVALID_TUPLE_INDEX_TYPE = 'Invalid tuple index type'
TUPLE_INDEX_OUT_OF_RANGE = 'Tuple index out of range'
ITERABLE_EXPECTED = 'Iterable expected'
INVALID_SLICE_INDEX = 'Slice index must be an integer or None'
CANNOT_INFER_LAMBDA_TYPE = 'Cannot infer type of lambda'
CANNOT_INFER_ITEM_TYPE = 'Cannot infer iterable item type'
CANNOT_ACCESS_INIT = 'Cannot access "__init__" directly'
CANNOT_ASSIGN_TO_METHOD = 'Cannot assign to a method'
CANNOT_ASSIGN_TO_TYPE = 'Cannot assign to a type'
INCONSISTENT_ABSTRACT_OVERLOAD = \
    'Overloaded method has both abstract and non-abstract variants'
READ_ONLY_PROPERTY_OVERRIDES_READ_WRITE = \
    'Read-only property cannot override read-write property'
FORMAT_REQUIRES_MAPPING = 'Format requires a mapping'
RETURN_TYPE_CANNOT_BE_CONTRAVARIANT = "Cannot use a contravariant type variable as return type"
FUNCTION_PARAMETER_CANNOT_BE_COVARIANT = "Cannot use a covariant type variable as a parameter"
INCOMPATIBLE_IMPORT_OF = "Incompatible import of"
FUNCTION_TYPE_EXPECTED = "Function is missing a type annotation"
ONLY_CLASS_APPLICATION = "Type application is only supported for generic classes"
RETURN_TYPE_EXPECTED = "Function is missing a return type annotation"
ARGUMENT_TYPE_EXPECTED = "Function is missing a type annotation for one or more arguments"
KEYWORD_ARGUMENT_REQUIRES_STR_KEY_TYPE = \
    'Keyword argument only valid with "str" key type in call to "dict"'
ALL_MUST_BE_SEQ_STR = 'Type of __all__ must be {}, not {}'
INVALID_TYPEDDICT_ARGS = \
    'Expected keyword arguments, {...}, or dict(...) in TypedDict constructor'
TYPEDDICT_KEY_MUST_BE_STRING_LITERAL = \
    'Expected TypedDict key to be string literal'
MALFORMED_ASSERT = 'Assertion is always true, perhaps remove parentheses?'
NON_BOOLEAN_IN_CONDITIONAL = 'Condition must be a boolean'
DUPLICATE_TYPE_SIGNATURES = 'Function has duplicate type signatures'
GENERIC_INSTANCE_VAR_CLASS_ACCESS = 'Access to generic instance variables via class is ambiguous'
CANNOT_ISINSTANCE_TYPEDDICT = 'Cannot use isinstance() with a TypedDict type'
CANNOT_ISINSTANCE_NEWTYPE = 'Cannot use isinstance() with a NewType type'
BARE_GENERIC = 'Missing type parameters for generic type'
IMPLICIT_GENERIC_ANY_BUILTIN = 'Implicit generic "Any". Use \'{}\' and specify generic parameters'
INCOMPATIBLE_TYPEVAR_VALUE = 'Value of type variable "{}" of {} cannot be {}'
UNSUPPORTED_ARGUMENT_2_FOR_SUPER = 'Unsupported argument 2 for "super"'






class PyccelError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super(PyccelException, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

class PyccelSyntaxError(Exception):
    pass

class PyccelSemanticError(Exception):
    pass


class ErrorInfo:
    """Representation of a single error message."""

    def __init__(self, filename,
                 line=None,
                 column=None,
                 severity=None,
                 message='',
                 symbol=None,
                 blocker=False):
        # The source file that was the source of this error.
        self.filename = filename
        # The line number related to this error within file.
        self.line = line
        # The column number related to this error with file.
        self.column = column
        # Either 'error', 'critical', or 'warning'.
        self.severity = severity
        # The error message.
        self.message = message
        # Symbol associated to the message
        self.symbol = symbol
        # If True, we should halt build after the file that generated this error.
        self.blocker = blocker

    def __str__(self):

#        from termcolor import colored, cprint
#        warning  = colored('warning', 'green', attrs=['reverse', 'blink'])
#        error = colored('error', 'red', attrs=['reverse', 'blink'])
#        critical = colored('critical', 'magenta', attrs=['reverse', 'blink'])
#
#        _severity_registry = {'error': error, 'critical': critical, 'warning':
#                              warning}

#        _severity_registry = {'error': 'E', 'critical': 'C', 'warning': 'W'}
        _severity_registry = {'error': 'error', 'critical': 'critical',
                              'warning': 'warning'}

        pattern = '|{severity}'
        text = pattern.format(severity=_severity_registry[self.severity])

        if self.line:
            if not self.column:
                text = '{text}: {line}'.format(text=text, line=self.line)
            else:
                text = '{text}: {line},{column}'.format(text=text, line=self.line,
                                                     column=self.column)

        text = '{text}| {msg}'.format(text=text, msg=self.message)

        if self.symbol:
            text = '{text} ({symbol})'.format(text=text, symbol=self.symbol)

        return text


class Errors:
    """Container for compile errors.
    """

    def __init__(self):
        self.error_info_map = None
        self._target = None
        self._parser_stage = None

        self.initialize()

    def initialize(self):
        self.error_info_map = OrderedDict()

        self._target = {}
        self._target['file'] = None
        self._target['module'] = None
        self._target['function'] = None
        self._target['class'] = None

    def reset(self):
        self.initialize()

    @property
    def parser_stage(self):
        return self._parser_stage

    def set_parser_stage(self, stage):
        assert(stage in ['syntax', 'semantic'])
        self._parser_stage = stage

    @property
    def target(self):
        return self._target

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
               blocker = False,
               severity = 'error',
               symbol = None,
               filename = None):
        """Report message at the given line using the current error context.
        stage: 'syntax' or 'semantic'
        """
        if filename is None:
            filename = self.target['file']

        info = ErrorInfo(filename,
                         line=line,
                         column=column,
                         severity=severity,
                         message=message,
                         symbol=symbol,
                         blocker=blocker)
        if blocker:
            if self.parser_stage == 'syntax':
                raise PyccelSyntaxError(str(info), '')
            elif self.parser_stage == 'semantic':
                raise PyccelSemanticError(str(info), '')
            # TODO what shall we do here?

        self.add_error_info(info)

    def _add_error_info(self, file, info):
        if file not in self.error_info_map:
            self.error_info_map[file] = []
        self.error_info_map[file].append(info)

    def add_error_info(self, info):
        self._add_error_info(info.filename, info)

    def num_messages(self):
        """Return the number of generated messages."""
        return sum(len(x) for x in self.error_info_map.values())

    def is_errors(self):
        """Are there any generated errors?"""
        return bool(self.error_info_map)

    def is_blockers(self):
        """Are the any errors that are blockers?"""
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
        for path in self.error_info_map.keys():
            errors = self.error_info_map[path]
            text += '>>> {path}\n'.format(path=path)
            for err in errors:
                text += str(err) + '\n'
        return text

if __name__ == '__main__':
    errors = Errors()
    errors.set_parser_stage('semantic')

    errors.set_target('script.py', 'file')

    errors.report(NO_RETURN_VALUE_EXPECTED, severity='warning', line=24)

    errors.set_target('make_knots', 'function')
    errors.report(INCOMPATIBLE_RETURN_VALUE_TYPE, symbol='y', severity='error',
                 line=34, column=17)
    errors.unset_target('function')

    print(errors)

    errors.set_target('eval_bsplines', 'function')
    errors.report(UNDEFINED_VARIABLE, symbol='x', severity='error', blocker=True)
    errors.unset_target('function')

