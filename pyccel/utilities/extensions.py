from pyccel.utilities.metaclasses import Singleton

class Extensions(metaclass=Singleton):
    def __init__(self):
        from pyccel.extensions.Openmp.extension import Openmp
        self._extensions = [Openmp()]

    def set_options(self, **options):
        for e in self._extensions:
            e.set_options(**options)

    def extend_syntax_parser(self, cls):
        extended = cls
        for e in self._extensions:
            extended = e.extend_syntax_parser(extended)
        return extended

    def extend_semantic_parser(self, cls):
        extended = cls
        for e in self._extensions:
            extended = e.extend_semantic_parser(extended)
        return extended

    def extend_printer(self, cls):
        extended = cls
        for e in self._extensions:
            extended = e.extend_printer(extended)
        return extended

class Extension():
    def extend_syntax_parser(self, cls):
        return cls

    def extend_semantic_parser(self, cls):
        return cls

    def extend_printer(self, cls):
        return cls

    def set_options(self):
        pass
