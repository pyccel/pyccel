from .metaclasses import Singleton

class PyccelStage(metaclass = Singleton):
    def __init__(self):
        self._stage = None

    def set_stage(self, stage):
        assert stage in ('syntactic', 'semantic', 'codegen')
        self._stage = stage

    def __eq__(self, other):
        return self._stage == other
