from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.strings import create_incremented_string
from .nameclashchecker import NameClashChecker

class PythonNameClashChecker(NameClashChecker, metaclass = Singleton):
    keywords = set()

    def has_clash(self, name, symbols):
        return any(name == k for k in self.keywords) or \
               any(name == s for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """ Get the name that will be used in the fortran code
        """
        prefix = name
        coll_symbols = self.keywords.copy()
        coll_symbols.update(s.lower() for s in symbols)
        if prefix in coll_symbols:
            counter = 1
            new_name, counter = create_incremented_string(coll_symbols,
                    prefix = prefix, counter = counter)
            name = name+new_name[-5:]
        return name

