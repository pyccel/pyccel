from pyccel.utilities.strings import create_incremented_string

class NameClashChecker:
    keywords = None

    def has_clash(self, name, symbols):
        name = name.lower()
        return any(name == k for k in self.keywords) or \
               any(name == s.lower() for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """ Get the name that will be used in the fortran code
        """
        prefix = name.lower()
        symbols = self.keywords.copy()
        symbols.update(s.lower() for s in symbols)
        if prefix in symbols:
            counter = 1
            new_name, counter = create_incremented_string(symbols,
                    prefix = prefix, counter = counter)
            name = name+new_name[-5:]
        return name
