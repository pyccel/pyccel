from .basic import PyccelAstNode

class FunctionDeclaration(PyccelAstNode):
    _my_attribute_nodes = ('_func', '_mod_var')
    def __init__(self, func, mod_var):
        self._func = func
        self._mod_var = mod_var
        super().__init__()

    @property
    def mod_var(self):
        return self._mod_var

    @property
    def function(self):
        return self._func
