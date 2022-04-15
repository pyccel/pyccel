import warnings

from pyccel.utilities.strings import create_incremented_string
from .nameclashchecker import NameClashChecker

class FortranNameClashChecker(NameClashChecker):
    # Keywords as mentioned on https://fortranwiki.org/fortran/show/Keywords
    keywords = set(['assign', 'backspace', 'block', 'blockdata',
            'call', 'close', 'common', 'continue', 'data',
            'dimension', 'do', 'else', 'elseif', 'end', 'endfile',
            'endif', 'endfunction', 'endmodule', 'endprogram',
            'endsubroutine', 'entry', 'equivalence', 'external',
            'format', 'function', 'goto', 'if', 'implicit',
            'intrinsic', 'open', 'parameter', 'pause', 'print',
            'program', 'read', 'return', 'rewind', 'rewrite',
            'save', 'stop', 'subroutine', 'then', 'write',
            'allocatable', 'allocate', 'case', 'contains', 'cycle',
            'deallocate', 'elsewhere', 'exit', 'include', 'interface',
            'intent', 'module', 'namelist', 'nullify', 'only',
            'operator', 'optional', 'pointer', 'private', 'procedure',
            'public', 'recursive', 'result', 'select', 'sequence',
            'target', 'use', 'while', 'where', 'elemental', 'forall',
            'pure', 'abstract', 'associate', 'asynchronous', 'bind',
            'class', 'deferred', 'enum', 'enumerator', 'extends',
            'final', 'flush', 'generic', 'import', 'non_overrideable',
            'nopass', 'pass', 'protected', 'value', 'volatile',
            'wait', 'codimension', 'concurrent', 'contiguous',
            'critical', 'error', 'submodule', 'sync', 'lock',
            'unlock', 'test'])

    def has_clash(self, name, symbols):
        name = name.lower()
        return any(name == k for k in self.keywords) or \
               any(name == s.lower() for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """ Get the name that will be used in the fortran code
        """
        if name[0] == '_':
            name = 'private'+name
        prefix = name.lower()
        coll_symbols = self.keywords.copy()
        coll_symbols.update(s.lower() for s in symbols)
        if prefix in coll_symbols:
            counter = 1
            new_name, counter = create_incremented_string(coll_symbols,
                    prefix = prefix, counter = counter)
            name = name+new_name[-5:]
        if len(name) > 96:
            warnings.warn("Name {} is too long for Fortran. This may cause compiler errors".format(name))
        return name
